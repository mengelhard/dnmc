import numpy as np

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.metrics import integrated_brier_score as ibs
from sksurv.metrics import brier_score as bs
from sksurv.util import Surv

from scipy.stats import chi2

from dataclasses import InitVar, dataclass, field


def evaluate_all(s_train, t_train, s_test, t_test, t_test_onehot, tp_onehot, bin_end_times,
    max_eval_bin):

    results = {}

    results['integrated_brier_score'] = integrated_brier_score(
        s_train, t_train, s_test, t_test,
        tp_onehot[:, :max_eval_bin],
        bin_end_times[:max_eval_bin])

    results['discrete_ci'] = discrete_concordance_index(s_test, t_test_onehot, tp_onehot)

    # Note: it's that last bin that's screwing up the d-calibration
    chisq, pval = d_calibration(
        s_test, t_test,
        tp_onehot[:, :max_eval_bin],
        bin_end_times[:max_eval_bin],
        bins=10)

    results['dcal_chisq'] = chisq
    results['dcal_pval'] = pval

    times, bs = brier_score(
        s_train, t_train, s_test, t_test,
        tp_onehot[:, :max_eval_bin],
        bin_end_times[:max_eval_bin])

    for t, b in zip(times, bs):
        results['brier_score_%i' % t] = b

    times, chisq, pval = one_calibration(
        s_test, t_test,
        tp_onehot[:, :max_eval_bin],
        bin_end_times[:max_eval_bin],
        n_cal_bins=10)

    for t, c, p in zip(times, chisq, pval):
        results['onecal_%i_chisq' % t] = c
        results['onecal_%i_pval' % t] = p

    times, ci = concordance_index(
        s_train, t_train, s_test, t_test,
        tp_onehot[:, :max_eval_bin],
        bin_end_times[:max_eval_bin])

    for t, c in zip(times, ci):
        results['ci_%i' % t] = c

    return results


def interpolate_linear(event_prob_by_bin, bin_end_times, event_times, return_survival=True):

    et = np.reshape(event_times, (-1))

    assert len(et) == 1 or len(et) == len(event_prob_by_bin)

    bin_lengths = np.diff([0] + list(bin_end_times))
    bin_start_times = np.array([0] + list(bin_end_times)[:-1])

    time_by_interval = (et[:, np.newaxis] - bin_start_times[np.newaxis, :])
    time_by_interval = time_by_interval / bin_lengths[np.newaxis, :]
    time_by_interval = np.minimum(time_by_interval, 1)
    time_by_interval = np.maximum(time_by_interval, 0)

    assert time_by_interval.max() <= 1
    assert time_by_interval.min() >= 0

    interpolated_cum_prob = np.sum(event_prob_by_bin * time_by_interval, axis=1)

    if return_survival:
        return 1 - interpolated_cum_prob
    else:
        return interpolated_cum_prob


def integrated_brier_score(s_train, t_train, s_test, t_test, tp_onehot, bin_end_times):
    '''
    tp_onehot is the probability of event occurrence in each bin
    surv_test_pred is the probability of remaining event free through the end of each bin
    bin_end_times gives the time corresponding to the end of each bin

    If tp_onehot contains more entries than bin_end_times, we assume that there is a final bin
    corresponding to event occurrence after the final bin (i.e. non-finite or beyond the horizon)
    '''

    N_bins = len(bin_end_times)
    N_pred = len(tp_onehot.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    #valid_idx = (t_test <= np.amax(t_train)) & (t_test >= np.amin(t_train))

    #surv_test_pred = 1 - np.cumsum(tp_onehot[valid_idx, :], axis=1)[:, :len(bin_end_times)]
    surv_test_pred = 1 - np.cumsum(tp_onehot, axis=1)[:, :len(bin_end_times)]

    surv_train = Surv().from_arrays(s_train, t_train)
    #surv_test = Surv().from_arrays(s_test[valid_idx], t_test[valid_idx])
    surv_test = Surv().from_arrays(s_test, t_test)

    return ibs(surv_train, surv_test, surv_test_pred, bin_end_times)


def brier_score(s_train, t_train, s_test, t_test, tp_onehot, bin_end_times):

    N_bins = len(bin_end_times)
    N_pred = len(tp_onehot.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    surv_test_pred = 1 - np.cumsum(tp_onehot, axis=1)[:, :len(bin_end_times)]

    surv_train = Surv().from_arrays(s_train, t_train)
    surv_test = Surv().from_arrays(s_test, t_test)
    
    return bs(surv_train, surv_test, surv_test_pred, bin_end_times)


def d_calibration(s_test, t_test, tp_onehot, bin_end_times, bins=10):

    # predictions are the survival probability at the event time (or censoring time)
    predictions = interpolate_linear(tp_onehot, bin_end_times, t_test, return_survival=True)

    event_indicators = s_test == 1

    # include minimum to catch if probability = 1.
    bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
    censored_bin_indexes = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]

    censored_predictions = predictions[~event_indicators]
    censored_contribution = 1 - (censored_bin_indexes / bins) * (
        1 / censored_predictions
    )
    censored_following_contribution = 1 / (bins * censored_predictions)

    contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

    following_contributions = np.matmul(
        censored_following_contribution, contribution_pattern[censored_bin_indexes]
    )
    single_contributions = np.matmul(
        censored_contribution, np.eye(bins)[censored_bin_indexes]
    )
    uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    bin_count = (
        single_contributions + following_contributions + uncensored_contributions
    )
    chi2_statistic = np.sum(
        np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
    )

    # return {
    #   'chisq_stat': chi2_statistic,
    #     'p_value': 1 - chi2.cdf(chi2_statistic, bins - 1),
    #     'bin_proportions': bin_count / len(predictions),
    #     'censored_contributions': (single_contributions + following_contributions) / len(predictions),
    #     'uncensored_contributions': uncensored_contributions / len(predictions),
    # }

    return (chi2_statistic, 1 - chi2.cdf(chi2_statistic, bins - 1))


def one_calibration(s_test, t_test, tp_onehot, bin_end_times, n_cal_bins=10, return_curves=False):

    N_bins = len(bin_end_times)
    N_pred = len(tp_onehot.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    cum_test_pred = np.cumsum(tp_onehot, axis=1)[:, :len(bin_end_times)]

    hs_stats = []
    p_vals = []
    times = []
    
    op = []
    ep = []

    for predictions, time in zip(cum_test_pred.T, bin_end_times):

        try:

            prediction_order = np.argsort(-predictions)
            predictions = predictions[prediction_order]
            event_times = t_test.copy()[prediction_order]
            event_indicators = (s_test == 1).copy()[prediction_order]

            # Can't do np.mean since split array may be of different sizes.
            binned_event_times = np.array_split(event_times, n_cal_bins)
            binned_event_indicators = np.array_split(event_indicators, n_cal_bins)
            probability_means = [np.mean(x) for x in np.array_split(predictions, n_cal_bins)]
            
            hosmer_lemeshow = 0
            
            observed_probabilities = []
            expected_probabilities = []
            
            for b in range(n_cal_bins):
                
                prob = probability_means[b]
                
                if prob == 1.0:
                    raise ValueError(
                        "One-Calibration is not well defined: the risk"
                        f"probability of the {b}th bin was {prob}."
                    )
                
                km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
                event_probability = 1 - km_model.predict(time)
                bin_count = len(binned_event_times[b])
                hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
                    bin_count * prob * (1 - prob)
                )
                
                observed_probabilities.append(event_probability)
                expected_probabilities.append(prob)

            hs_stats.append(hosmer_lemeshow)
            p_vals.append(1 - chi2.cdf(hosmer_lemeshow, n_cal_bins - 1))
            times.append(time)
            
            op.append(observed_probabilities)
            ep.append(expected_probabilities)

            # return dict(
            #     p_value=1 - chi2.cdf(hosmer_lemeshow, bins - 1),
            #     observed=observed_probabilities,
            #     expected=expected_probabilities,
            # )

        except Exception as e:

            print('Failed for time', time)
            print(e)
            
    if return_curves:
        return np.array(times), np.array(hs_stats), np.array(p_vals), np.array(op), np.array(ep)

    return np.array(times), np.array(hs_stats), np.array(p_vals)


def s_cal(event_indicator, observed_time, predicted_cum_event_prob, time_of_prediction, calibration_bw=.1, stride=1):
            
    num_events = len(event_indicator)

    prediction_order = np.argsort(-predicted_cum_event_prob)
    sorted_predictions = predicted_cum_event_prob[prediction_order]
    sorted_times = observed_time[prediction_order]
    sorted_event_indicators = event_indicator[prediction_order]

    num_events_in_window = int(num_events * calibration_bw)

    start_indices = np.arange(0, num_events - num_events_in_window, stride)

    pp = []
    op = []

    for idx in start_indices:

        avg_prob = np.mean(sorted_predictions[idx: idx + num_events_in_window])

        km_model = KaplanMeier(
            sorted_times[idx: idx + num_events_in_window],
            sorted_event_indicators[idx: idx + num_events_in_window]
        )

        event_prob = 1 - km_model.predict(time_of_prediction)

        pp.append(avg_prob)
        op.append(event_prob)
        
    return pp, op


def concordance_index(s_train, t_train, s_test, t_test, tp_onehot, bin_end_times,
    ipcw=True, **kwargs):

    if ipcw:
        surv_train = Surv().from_arrays(s_train, t_train)
        surv_test = Surv().from_arrays(s_test, t_test)

    N_bins = len(bin_end_times)
    N_pred = len(tp_onehot.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    cum_test_pred = np.cumsum(tp_onehot, axis=1)[:, :len(bin_end_times)]

    results = []
    times = []

    for estimate, time in zip(cum_test_pred.T, bin_end_times):

        try:

            if ipcw:
                results.append(concordance_index_ipcw(
                    surv_train, surv_test, estimate, tau=time, **kwargs)[0])
            else:
                results.append(concordance_index_censored(
                    s_test, t_test, estimate, **kwargs)[0])

            times.append(time)

        except Exception as e:

            print('Failed for time', time)
            print(e)

    return np.array(times), np.array(results)


def discrete_concordance_index(s_test, t_test_onehot, tp_onehot):

    s_true = np.array(s_test).copy()
    t_true = np.array(t_test_onehot).copy()
    t_pred = np.array(tp_onehot).copy()

    t_true_idx = np.argmax(t_true, axis=1)
    t_pred_cdf = np.cumsum(t_pred, axis=1)

    concordant = 0
    total = 0

    N = len(s_true)
    idx = np.arange(N)

    for i in range(N):

        if s_true[i] == 0:
            continue

        # time bucket of observation for i, then for all but i
        tti_idx = t_true_idx[i]
        tt_idx = t_true_idx[idx != i]

        # calculate predicted risk for i at the time of their event
        tpi = t_pred_cdf[i, tti_idx]

        # predicted risk at that time for all but i
        tp = t_pred_cdf[idx != i, tti_idx]

        total += np.sum(tti_idx < tt_idx) # observed in i first
        concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

    return concordant / total


@dataclass
class KaplanMeier:
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (
            self.area_times[censor_indexes] - censor_times
        ) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob

