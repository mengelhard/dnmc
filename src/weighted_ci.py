import numpy as np


def main():

	N = 10000

	s_true = np.random.rand(N) > .5
	t_true = np.exp(np.random.randn(N))
	c_pred = np.random.rand(N)
	t_pred = np.exp(np.random.randn(N))

	print('Random guessing CI = %.2f' % weighted_ci(s_true, t_true, t_pred, weighted=True))
	print('(unweighted) Random guessing CI = %.2f' % weighted_ci(s_true, t_true, t_pred))

	s_true = [1, 0, 1]
	t_true = [1.1, 5.1, 3.1]
	c_pred = [.8, .2, .7]
	t_pred = [1, 3, 4]

	print('Decent ordering CI = %.2f' % weighted_ci(s_true, t_true, t_pred, weighted=True))
	print('(unweighted) Decent ordering CI = %.2f' % weighted_ci(s_true, t_true, t_pred))


def weighted_ci(st, tt, tp, weighted=False, min_t=0.):

	valid = tt > min_t

	s_true = np.array(st).copy()[valid]
	t_true = np.array(tt).copy()[valid]
	t_pred = np.array(tp).copy()[valid]

	if weighted:

		sorted_events, sorted_estimate, _ = nelson_aalen(s_true, t_true)
		p_later_sorted = (sorted_estimate[1] - sorted_estimate) / (1 - sorted_estimate)
		p_later_dict = {k: v for k, v in zip(sorted_events, p_later_sorted)}
		weight = np.maximum([p_later_dict[t] for t in t_true], s_true)

	else:

		weight = np.ones_like(s_true)

	concordant = 0
	total = 0

	N = len(s_true)
	idx = np.arange(N)

	for i in range(N):

		sti = s_true[i]

		if sti == 0:
			continue

		tti = t_true[i]
		tpi = t_pred[i]

		st = s_true[idx != i]
		tt = t_true[idx != i]
		tp = t_pred[idx != i]
		w = weight[idx != i]

		total += np.sum(w * (tti < tt))
		concordant += np.sum(w * (tti < tt) * (tpi < tp))

	return concordant / total


def discrete_ci(st, tt, tp):

	s_true = np.array(st).copy()
	t_true = np.array(tt).copy()
	t_pred = np.array(tp).copy()

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
		tp = t_pred_cdf[i, idx != tti_idx]

		total += np.sum(tti_idx < tt_idx) # observed in i first
		concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

	return concordant / total


def nelson_aalen(s_true, t_true):

	t_sortedorder = np.argsort(t_true)
	t_sorted = t_true[t_sortedorder]
	s_sorted = s_true[t_sortedorder]
	
	estimate = []
	variance = []

	for s, t in zip(s_sorted, t_sorted):
		n = np.sum(t_true >= t)
		if s == 1:
			estimate.append(1 / n)
			variance.append((n - 1) / ((n - 1) * (n ** 2)))
		else:
			estimate.append(0)
			variance.append(0)
		
	return t_sorted, np.cumsum(estimate), np.cumsum(variance)


if __name__ == '__main__':
	main()