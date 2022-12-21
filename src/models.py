import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from sklearn.metrics import roc_auc_score

## FEATURES TO ADD
## - model should follow normal survival analysis conventions
## - pass s and t, and binning should be internal
## - survival function should linearly interpolate
## --- will then need the actual times to evaluate survival function
## - create functions to predict CMF and survival function at a given time

class DNMC(Model):
    
    def __init__(self, bins,
                 phi_layer_sizes=[256,], psi_layer_sizes=[256,], omega_layer_sizes=[256,],
                 e_layer_sizes=[256,], t_layer_sizes=[256,], c_layer_sizes=[256,],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 include_psi=True,
                 activation='relu',
                 rep_activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-8):
        
        super(DNMC, self).__init__()

        self.bins = bins
        self.n_bins = len(bins) - 1
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.include_psi = include_psi
        self.activation = activation
        self.tol = tol
        
        self.phi_layers = [self.dense(ls, activation=rep_activation) for ls in phi_layer_sizes]
        if include_psi:
            self.psi_layers = [self.dense(ls, activation=rep_activation) for ls in psi_layer_sizes]
        self.omega_layers = [self.dense(ls, activation=rep_activation) for ls in omega_layer_sizes]
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [self.dense(1, activation='sigmoid')]
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
        
        self.phi_model = Sequential(self.phi_layers)
        if include_psi:
            self.psi_model = Sequential(self.psi_layers)
        self.omega_model = Sequential(self.omega_layers)
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:

            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [Sequential(cl) for cl in self.c_layers]

            else:
            
                self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        if self.include_psi:
            self.psi = self.psi_model(x)
        self.phi = self.phi_model(x)
        self.omega = self.omega_model(x)
        
        if self.include_psi:
            self.e_pred = tf.squeeze(self.e_model(tf.concat([self.phi, self.psi], axis=-1)), axis=1)
            self.t_pred = self.t_model(tf.concat([self.psi, self.omega], axis=-1))
        else:
            self.e_pred = tf.squeeze(self.e_model(self.phi), axis=1)
            self.t_pred = self.t_model(self.omega)
        
        if self.include_censoring_density:
            if self.dependent_censoring:
                if self.include_psi:
                    self.c_pred = [
                        cm(tf.concat([self.psi, self.omega], axis=-1))
                        for cm in self.c_model
                    ]
                else:
                    self.c_pred = [
                        cm(self.omega)
                        for cm in self.c_model
                    ]
                return self.e_pred, self.t_pred, self.c_pred
            else:
                if self.include_psi:
                    self.c_pred = self.c_model(tf.concat([self.psi, self.omega], axis=-1))
                else:
                    self.c_pred = self.c_model(self.omega)
                return self.e_pred, self.t_pred, self.c_pred

        else:
            return self.e_pred, self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)


    def predict(self, x):
        return self.forward_pass(x)


    def predict_survival_function(self, x, t):
        y_complete_bins = get_proportion_of_bins_completed(np.ones(len(x)) * t, self.bins)
        return 1 - tf.reduce_sum(y_complete_bins * self.forward_pass(x), axis=1) + self.tol
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, t, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, t, s))

        # MMD Term
        # l = nll + self.ld * tf.cast(mmd(x, s), dtype=tf.float32)
        l = nll + self.ld * tf.cast(mmd(self.omega_model(x), s), dtype=tf.float32)

        # Global L2 regularizer
        l += tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, t, s):

        y = discretize_times(t, self.bins)
        yt = tf.cast(y, dtype=tf.float32)

        y_complete_bins = tf.cast(
            get_proportion_of_bins_completed(t, self.bins),
            dtype=tf.float32
        )
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)            

            if self.dependent_censoring:

                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred[1], axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred[1]), axis=1) + self.tol

            else:
            
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred, axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = 1 - tf.reduce_sum(y_complete_bins * t_pred, axis=1) + self.tol
        #Ft = tf.reduce_sum(yt * self.survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


class NMC(Model):
    
    def __init__(self, bins,
                 e_layer_sizes=[256, 256], t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NMC, self).__init__()

        self.bins = bins
        self.n_bins = len(bins) - 1
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.activation = activation
        self.tol = tol
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [self.dense(1, activation='sigmoid')]
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:
            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [
                    Sequential(cl)
                    for cl in self.c_layers
                ]
            else:
                self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.e_pred = tf.squeeze(self.e_model(x), axis=1)
        self.t_pred = self.t_model(x)

        if self.include_censoring_density:
            if self.dependent_censoring:
                self.c_pred = [cm(x) for cm in self.c_model]
            else:
                self.c_pred = self.c_model(x)
            return self.e_pred, self.t_pred, self.c_pred

        else:
            return self.e_pred, self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)


    def predict(self, x):
        return self.forward_pass(x)


    def predict_survival_function(self, x, t):
        y_complete_bins = get_proportion_of_bins_completed(np.ones(len(x)) * t, self.bins)
        return 1 - tf.reduce_sum(y_complete_bins * self.forward_pass(x), axis=1) + self.tol
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, t, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, t, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    

    def nll(self, x, t, s):

        y = discretize_times(t, self.bins)
        yt = tf.cast(y, dtype=tf.float32)

        y_complete_bins = tf.cast(
            get_proportion_of_bins_completed(t, self.bins),
            dtype=tf.float32
        )
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)

            if self.dependent_censoring:
                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred[1], axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred[1]), axis=1) + self.tol
            else:
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred, axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = 1 - tf.reduce_sum(y_complete_bins * t_pred, axis=1) + self.tol
        #Ft = tf.reduce_sum(yt * self.survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


class NSurv(Model):
    
    def __init__(self, bins,
                 t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NSurv, self).__init__()

        self.bins = bins
        self.n_bins = len(bins) - 1
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.activation=activation
        self.tol = tol
        
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
        self.t_model = Sequential(self.t_layers)

        if self.include_censoring_density:
            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [
                    Sequential(cl)
                    for cl in self.c_layers
                ]
            else:
                self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [self.dense(self.n_bins, activation='softmax')]
                self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.t_pred = self.t_model(x)

        if self.include_censoring_density:
            if self.dependent_censoring:
                self.c_pred = [cm(x) for cm in self.c_model]
            else:
                self.c_pred = self.c_model(x)
            return self.t_pred, self.c_pred

        else:
            return self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)


    def predict(self, x):
        return self.forward_pass(x)


    def predict_survival_function(self, x, t):
        y_complete_bins = get_proportion_of_bins_completed(np.ones(len(x)) * t, self.bins)
        return 1 - tf.reduce_sum(y_complete_bins * self.forward_pass(x), axis=1) + self.tol
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, t, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, t, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, t, s):

        y = discretize_times(t, self.bins)        
        yt = tf.cast(y, dtype=tf.float32)

        y_complete_bins = tf.cast(
            get_proportion_of_bins_completed(t, self.bins),
            dtype=tf.float32
        )

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = 1 - tf.reduce_sum(y_complete_bins * t_pred, axis=1) + self.tol

        # removed previous version which was a step rather than linear approx to survival function
        # Ft = tf.reduce_sum(yt * survival_from_density(t_pred)[:, :-1], axis=1) + self.tol

        y = discretize_times(t, self.bins)
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            t_pred, c_pred = self.forward_pass(x)

            if self.dependent_censoring:
                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred[1], axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred[1]), axis=1) + self.tol
            else:
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = 1 - tf.reduce_sum(y_complete_bins * c_pred, axis=1) + self.tol
                #Fc = tf.reduce_sum(yt * self.survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = 1 - tf.reduce_sum(y_complete_bins * t_pred, axis=1) + self.tol
        #Ft = tf.reduce_sum(yt * self.survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(Ft) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


class NSurv_MMD(Model):
    
    def __init__(self, bins,
                 encoder_layer_sizes=[256, ],
                 decoder_layer_sizes=[256, ],
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NSurv_MMD, self).__init__()

        self.bins = bins
        self.n_bins = len(bins) - 1
        
        self.ld = ld
        self.lr = lr
        
        self.activation=activation
        self.tol = tol
        
        self.encoder_layers = [self.dense(ls) for ls in encoder_layer_sizes]
        self.decoder_layers = [self.dense(ls) for ls in decoder_layer_sizes] + [self.dense(self.n_bins + 1, activation='softmax')]
        
        self.encoder = Sequential(self.encoder_layers)
        self.decoder = Sequential(self.decoder_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.representation = self.encoder(x)
        self.t_pred = self.decoder(self.representation)
        
        return self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)


    def predict(self, x):
        return self.forward_pass(x)


    def predict_survival_function(self, x, t):
        y_complete_bins = get_proportion_of_bins_completed(np.ones(len(x)) * t, self.bins)
        return 1 - tf.reduce_sum(y_complete_bins * self.forward_pass(x)[:, :-1], axis=1) + self.tol

    
    def loss(self, x, t, s, mmd_binary_variable):
        
        nll = tf.reduce_mean(self.nll(x, t, s))
        l = nll + tf.reduce_sum(self.losses)
        
        # MMD Term
        l += self.ld * tf.cast(mmd(self.representation, mmd_binary_variable), dtype=tf.float32)
        
        return l, nll
    
    
    def nll(self, x, t, s):

        y = discretize_times(t, self.bins)        
        yt = tf.cast(y, dtype=tf.float32)

        y_complete_bins = tf.cast(
            get_proportion_of_bins_completed(t, self.bins),
            dtype=tf.float32
        )

        t_pred = self.forward_pass(x)

        ft = tf.reduce_sum(yt * t_pred[:, :-1], axis=1) + self.tol
        Ft = 1 - tf.reduce_sum(y_complete_bins * t_pred[:, :-1], axis=1) + self.tol

        # removed previous version which was a step rather than linear approx to survival function
        # Ft = tf.reduce_sum(yt * survival_from_density(t_pred)[:, :-1], axis=1) + self.tol
        
        ll1 = tf.math.log(ft)
        ll2 = tf.math.log(Ft)
        
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


class MLP(Model):
    
    def __init__(self,
                 e_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 activation='relu',
                 ld=1e-3, lr=1e-3):
        
        super(MLP, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.activation=activation
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        self.e_model = Sequential(self.e_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        self.e_pred = tf.squeeze(self.e_model(x), axis=1)
        return self.e_pred
    
    
    def call(self, x):
        return self.forward_pass(x)
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, t, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, t, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, t, s):
        
        e_pred = self.forward_pass(x)
        
        l1 = e_pred
        l2 = (1 - e_pred)
            
        ll = tf.cast(s, dtype=tf.float32) * tf.math.log(l1)
        ll += tf.cast((1 - s), dtype=tf.float32) * tf.math.log(l2)
        
        return -1 * ll


def discretize_times(times, bins):

    bin_starts = np.array(bins[:-1])[np.newaxis, :]
    bin_ends = np.array(bins[1:])[np.newaxis, :]

    t = np.array(times)[:, np.newaxis]

    return ((t > bin_starts) & (t <= bin_ends)).astype(float)


def get_proportion_of_bins_completed(times, bins):

    bin_starts = np.array(bins[:-1])[np.newaxis, :]
    bin_ends = np.array(bins[1:])[np.newaxis, :]

    bin_lengths = bin_ends - bin_starts

    t = np.array(times)[:, np.newaxis]

    return np.maximum(np.minimum((t - bin_starts) / bin_lengths, 1), 0)


def mmd(x, s, beta=None):

    if beta is None:
        beta = get_median(tf.reduce_sum((x[:, tf.newaxis, :] - x[tf.newaxis, :, :]) ** 2, axis=-1))
    
    x0 = tf.boolean_mask(x, s == 0, axis=0)
    x1 = tf.boolean_mask(x, s == 1, axis=0)

    x0x0 = gaussian_kernel(x0, x0, beta)
    x0x1 = gaussian_kernel(x0, x1, beta)
    x1x1 = gaussian_kernel(x1, x1, beta)
    
    return tf.reduce_mean(x0x0) - 2. * tf.reduce_mean(x0x1) + tf.reduce_mean(x1x1)


def gaussian_kernel(x1, x2, beta=1.):
    return tf.exp(-1. * beta * tf.reduce_sum((x1[:, tf.newaxis, :] - x2[tf.newaxis, :, :]) ** 2, axis=-1))


def survival_from_density(f):
    return 1 - tf.math.cumsum(f, axis=1)
    #return tf.math.cumsum(f, reverse=True, axis=1)


def train_model(
    model, train_data, val_data, n_epochs,
    batch_size=50, learning_rate=1e-3, early_stopping_criterion=2,
    overwrite_output=True):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #@tf.function
    def train_step(x, t, s):
        with tf.GradientTape() as tape:
            train_loss, train_nll = model.loss(x, t, s)
            #print(train_loss, train_nll)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss, train_nll

    #@tf.function
    def test_step(x, t, s):
        val_loss, val_nll = model.loss(x, t, s)
        return val_loss, val_nll
    
    best_val_loss = np.inf
    no_decrease = 0

    for epoch_idx in range(n_epochs):

        train_losses = []
        train_nlls = []

        for batch_idx, (xt, tt, st) in enumerate(get_batches(*train_data, batch_size=batch_size)):

            train_loss, train_nll = train_step(xt, tt, st)

            train_losses.append(train_loss)
            train_nlls.append(train_nll)

        # Display metrics at the end of each epoch.
        #print('Epoch training loss: %.4f, NLL = %.4f' % (np.mean(batch_losses), np.mean(batch_nll)))

        val_losses = []
        val_nlls = []

        # Run a validation loop at the end of each epoch.
        for batch_idx, (xv, tv, sv) in enumerate(get_batches(*val_data, batch_size=batch_size)):

            val_loss, val_nll = test_step(xv, tv, sv)

            val_losses.append(val_loss)
            val_nlls.append(val_nll)
            
        new_val_loss = np.mean(val_losses)

        if overwrite_output:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls)),
                end='\r'
            )

        else:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
            )
                
        if new_val_loss > best_val_loss:
            no_decrease += 1
        else:
            no_decrease = 0
            best_val_loss = new_val_loss
            
        if no_decrease == early_stopping_criterion:
            break

    if overwrite_output:
        print(
            'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
            % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
        )
        print('')


def train_model_MMD(
    model, train_data, val_data, n_epochs,
    batch_size=50, learning_rate=1e-3, early_stopping_criterion=2,
    overwrite_output=True):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #@tf.function
    def train_step(x, t, s, mbv):
        with tf.GradientTape() as tape:
            train_loss, train_nll = model.loss(x, t, s, mbv)
            #print(train_loss, train_nll)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss, train_nll

    #@tf.function
    def test_step(x, t, s, mbv):
        val_loss, val_nll = model.loss(x, t, s, mbv)
        return val_loss, val_nll
    
    best_val_loss = np.inf
    no_decrease = 0

    for epoch_idx in range(n_epochs):

        train_losses = []
        train_nlls = []

        for batch_idx, (xt, tt, st, mbvt) in enumerate(get_batches(*train_data, batch_size=batch_size)):

            train_loss, train_nll = train_step(xt, tt, st, mbvt)

            train_losses.append(train_loss)
            train_nlls.append(train_nll)

        # Display metrics at the end of each epoch.
        #print('Epoch training loss: %.4f, NLL = %.4f' % (np.mean(batch_losses), np.mean(batch_nll)))

        val_losses = []
        val_nlls = []

        # Run a validation loop at the end of each epoch.
        for batch_idx, (xv, tv, sv, mbvv) in enumerate(get_batches(*val_data, batch_size=batch_size)):

            val_loss, val_nll = test_step(xv, tv, sv, mbvv)

            val_losses.append(val_loss)
            val_nlls.append(val_nll)
            
        new_val_loss = np.mean(val_losses)

        if overwrite_output:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls)),
                end='\r'
            )

        else:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
            )
                
        if new_val_loss > best_val_loss:
            no_decrease += 1
        else:
            no_decrease = 0
            best_val_loss = new_val_loss
            
        if no_decrease == early_stopping_criterion:
            break

    if overwrite_output:
        print(
            'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
            % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
        )
        print('')


def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)


def onehot(arr, ncategories=None):
    if ncategories is None:
        ncategories = len(np.unique(arr))
    return np.eye(ncategories)[arr.astype(int)]


def get_batches(*arrs, batch_size=1):
    l = len(arrs[0])
    for ndx in range(0, l, batch_size):
        yield (arr[ndx:min(ndx + batch_size, l)] for arr in arrs)


def pull_split_max_avg(layer, indices):
    weights = layer.get_weights()[0]
    if len(indices) > 0:
        in_avg = np.amax(np.abs(weights[indices, :]), axis=1).mean()
    else:
        in_avg = 0.
    if len(indices) < len(weights):
        out_avg = np.amax(np.abs(np.delete(weights, indices, axis=0)), axis=1).mean()
    else:
        out_avg = 0.
    return in_avg, out_avg


def pull_split_avg(layer, indices):
    weights = layer.get_weights()[0]
    if len(indices) > 0:
        in_avg = np.abs(weights[indices, :]).mean()
    else:
        in_avg = 0.
    if len(indices) < len(weights):
        out_avg = np.abs(np.delete(weights, indices, axis=0)).mean()
    else:
        out_avg = 0.
    return in_avg, out_avg

