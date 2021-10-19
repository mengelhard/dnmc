import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(2021)

from models import DNMC, NMC, NSurv, MLP, train_model, evaluate_model

df = pd.read_csv('http://pssp.srv.ualberta.ca/system/predictors/datasets/000/000/032/original/All_Data_updated_may2011_CLEANED.csv?1350302245')

numrc_cols = df.nunique() > 2
df.loc[:, numrc_cols] = (df.loc[:, numrc_cols] - df.loc[:, numrc_cols].mean()) / df.loc[:, numrc_cols].std()

OUTCOMES = ['SURVIVAL', 'CENSORED']
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
DATATYPE = 'synth_censoring'
RESULTS_NAME = '../results/NACD_' + DATATYPE + '.csv'

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
                e_prob_spread=3.)

        elif DATATYPE == 'synth_censoring':

            synth = generate_synth_censoring(
                X, df['SURVIVAL'].values, 1 - df['CENSORED'].values,
                num_distinct, N_BINS, random_state,
                e_prob_spread=3.)
        
        x_train, x_val, x_test = X[:1500], X[1500:1900], X[1900:]

        y = onehot(synth['y_disc'], ncategories=10)
        y_train, y_val, y_test = y[:1500], y[1500:1900], y[1900:]

        s_train, s_val, s_test = synth['s'][:1500], synth['s'][1500:1900], synth['s'][1900:]
        e_train, e_val, e_test = synth['e'][:1500], synth['e'][1500:1900], synth['e'][1900:]
    
    #for lr in np.logspace(-2, -1, 6):
        
        # Run NMC
        
        print('Running NMC with lr =', lr)
        
        model = NMC(n_bins=N_BINS, lr=lr)
        
        train_model(
            model, (x_train, y_train, s_train), (x_val, y_val, s_val),
            MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        
        all_results.append(
            evaluate_model(
                model, (x_test, y_test, s_test), e_test,
                (synth['shared_features'], synth['tc_features'], synth['e_features']),
                dataset='nacd', random_state=random_state))
        
        # Run NSurv
        
        print('Running NSurv with lr =', lr)
        
        model = NSurv(n_bins=N_BINS, lr=lr)
        
        train_model(
            model, (x_train, y_train, s_train), (x_val, y_val, s_val),
            MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        
        all_results.append(
            evaluate_model(
                model, (x_test, y_test, s_test), e_test,
                (synth['shared_features'], synth['tc_features'], synth['e_features']),
                dataset='nacd', random_state=random_state))
        
        # Run MLP
        
        print('Running MLP with lr =', lr)
        
        model = MLP(lr=lr)
        
        train_model(
            model, (x_train, y_train, s_train), (x_val, y_val, s_val),
            MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        
        all_results.append(
            evaluate_model(
                model, (x_test, y_test, s_test), e_test,
                (synth['shared_features'], synth['tc_features'], synth['e_features']),
                dataset='nacd', random_state=random_state))
        
        # Run DNMC
        
        for ld in [1., 10.]:
            
            print('Running DNMC (with Psi) with lr =', lr, 'and ld =', ld)
            
            model = DNMC(n_bins=N_BINS, lr=lr, ld=ld)
        
            train_model(
                model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

            all_results.append(
                evaluate_model(
                    model, (x_test, y_test, s_test), e_test,
                    (synth['shared_features'], synth['tc_features'], synth['e_features']),
                    dataset='nacd', random_state=random_state))

            print('Running DNMC (NO Psi) with lr =', lr, 'and ld =', ld)
            
            model = DNMC(n_bins=N_BINS, lr=lr, ld=ld, include_psi=False)
        
            train_model(
                model, (x_train, y_train, s_train), (x_val, y_val, s_val),
                MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

            results = evaluate_model(
                model, (x_test, y_test, s_test), e_test,
                (synth['shared_features'], synth['tc_features'], synth['e_features']),
                dataset='nacd', random_state=random_state)

            results['model'] = 'DNMC_noPsi'

            all_results.append(results)

        pd.DataFrame(all_results).to_csv(RESULTS_NAME)
