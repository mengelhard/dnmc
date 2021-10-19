import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(2021)

from models import DNMC, NMC, NSurv, MLP, train_model, evaluate_model

FILL_VALUES = {
    'alb': 3.5,
    'pafi': 333.3,
    'bili': 1.01,
    'crea': 1.01,
    'bun': 6.51,
    'wblc': 9.,
    'urine': 2502.
}

TO_DROP = ['aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday']
TO_DROP = TO_DROP + ['sfdm2', 'hospdead']

# load, drop columns, fill using specified fill values
df = pd.read_csv('../datasets/support2.csv').drop(TO_DROP,axis=1).fillna(value=FILL_VALUES)

# get dummies for categorical vars
df = pd.get_dummies(df, dummy_na=True)

# fill remaining values to the median

df = df.fillna(df.median())

# standardize numeric columns

numrc_cols = df.dtypes == 'float64'
df.loc[:, numrc_cols] = (df.loc[:, numrc_cols] - df.loc[:, numrc_cols].mean()) / df.loc[:, numrc_cols].std()

OUTCOMES = ['death', 'd.time']
X = df.drop(OUTCOMES, axis=1).sample(frac=1, random_state=2021)
X = X.values

print('There are', X.shape[1], 'features')

from generate_data import generate_semi_synthetic, generate_synth_censoring, onehot

### BEGIN COLLECTING RESULTS HERE ###
all_results = []
all_weight_results = []

LEARNING_RATE = 1e-3
BATCH_SIZE = 100
N_BINS = 10
MAX_EPOCHS = 500
lr = 0.03
DATATYPE = 'synthetic'
RESULTS_NAME = '../results/SUPPORT_' + DATATYPE + '.csv'

assert DATATYPE in ['synth_censoring', 'synthetic', 'real']

# NOTE that we are skipping importance weights here.

for random_state in [2020, 2016, 2013]:

    for num_distinct in [4, 8, 12, 16]:

        num_shared = 20 - num_distinct
    
        print('')
        print('Starting runs with random state', random_state, 'and %i distinct features' % num_distinct)
        print('')

        if DATATYPE == 'synthetic':

            synth = generate_semi_synthetic(
                X, num_distinct, num_shared, N_BINS, random_state,
                e_prob_spread=2.)

        elif DATATYPE == 'synth_censoring':

            synth = generate_synth_censoring(
                X, df['d.time'].values, df['death'].values,
                num_distinct, N_BINS, random_state,
                e_prob_spread=2.)
        
        x_train, x_val, x_test = X[:6000], X[6000:7500], X[7500:]

        y = onehot(synth['y_disc'], ncategories=10)
        y_train, y_val, y_test = y[:6000], y[6000:7500], y[7500:]

        s_train, s_val, s_test = synth['s'][:6000], synth['s'][6000:7500], synth['s'][7500:]
        e_train, e_val, e_test = synth['e'][:6000], synth['e'][6000:7500], synth['e'][7500:]
    
    #for lr in np.logspace(-2, -1, 6):
        
        # Run NMC
        
        print('Running NMC with lr =', lr)
        
        model = NMC(n_bins=N_BINS, lr=lr)

        try:
        
            train_model(
                model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
            
            all_results.append(
                evaluate_model(
                    model, (x_test, y_test, s_test), e_test,
                    (synth['shared_features'], synth['tc_features'], synth['e_features']),
                    dataset='nacd', random_state=random_state))

        except:

            print('Run Failed')
        
        # Run NSurv
        
        print('Running NSurv with lr =', lr)
        
        model = NSurv(n_bins=N_BINS, lr=lr)

        try:
        
            train_model(
                model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
            
            all_results.append(
                evaluate_model(
                    model, (x_test, y_test, s_test), e_test,
                    (synth['shared_features'], synth['tc_features'], synth['e_features']),
                    dataset='nacd', random_state=random_state))

        except:

            print('Run Failed')
        
        # Run MLP
        
        print('Running MLP with lr =', lr)
        
        model = MLP(lr=lr)

        try:
        
            train_model(
                model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
            
            all_results.append(
                evaluate_model(
                    model, (x_test, y_test, s_test), e_test,
                    (synth['shared_features'], synth['tc_features'], synth['e_features']),
                    dataset='nacd', random_state=random_state))

        except:

            print('Run Failed')
        
        # Run DNMC
        
        for ld in [1., 10.]:
            
            print('Running DNMC (with Psi) with lr =', lr, 'and ld =', ld)
            
            model = DNMC(n_bins=N_BINS, lr=lr, ld=ld)

            try:
        
                train_model(
                    model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                    MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

                all_results.append(
                    evaluate_model(
                        model, (x_test, y_test, s_test), e_test,
                        (synth['shared_features'], synth['tc_features'], synth['e_features']),
                        dataset='nacd', random_state=random_state))

            except:

                print('Run Failed')

            print('Running DNMC (NO Psi) with lr =', lr, 'and ld =', ld)
            
            model = DNMC(n_bins=N_BINS, lr=lr, ld=ld, include_psi=False)

            try:
        
                train_model(
                    model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                    MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

                results = evaluate_model(
                    model, (x_test, y_test, s_test), e_test,
                    (synth['shared_features'], synth['tc_features'], synth['e_features']),
                    dataset='nacd', random_state=random_state)

                results['model'] = 'DNMC_noPsi'

                all_results.append(results)

            except:

                print('Run Failed')

        pd.DataFrame(all_results).to_csv(RESULTS_NAME)
