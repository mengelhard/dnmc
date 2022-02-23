import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from sklearn.metrics import roc_auc_score


class DNMC(Model):
    
    def __init__(self,
                 phi_layer_sizes=[256,], psi_layer_sizes=[256,], omega_layer_sizes=[256,],
                 e_layer_sizes=[256,], t_layer_sizes=[256,], c_layer_sizes=[256,],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 include_psi=True,
                 mtlr_head=False,
                 n_bins=50,
                 activation='relu',
                 rep_activation='relu',
                 ld=1e-3, lr=1e-3, l_mtlr=0., tol=1e-8):
        
        super(DNMC, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.l_mtlr = l_mtlr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.include_psi = include_psi
        self.mtlr_head = mtlr_head
        self.n_bins = n_bins
        self.activation = activation
        self.tol = tol
        
        self.phi_layers = [self.dense(ls, activation=rep_activation) for ls in phi_layer_sizes]
        if include_psi:
            self.psi_layers = [self.dense(ls, activation=rep_activation) for ls in psi_layer_sizes]
        self.omega_layers = [self.dense(ls, activation=rep_activation) for ls in omega_layer_sizes]
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        if mtlr_head:
            self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins - 1, activation='sigmoid')]
        else:
            self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        
        self.phi_model = Sequential(self.phi_layers)
        if include_psi:
            self.psi_model = Sequential(self.psi_layers)
        self.omega_model = Sequential(self.omega_layers)
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:

            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [Sequential(cl) for cl in self.c_layers]

            else:
            
                if mtlr_head:
                    self.t_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins - 1, activation='sigmoid')]
                else:
                    self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
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
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))

        # MMD Term
        # l = nll + self.ld * tf.cast(self.mmd(x, s), dtype=tf.float32)
        l = nll + self.ld * tf.cast(self.mmd(self.omega_model(x), s), dtype=tf.float32)

        # Global L2 regularizer
        l += tf.reduce_sum(self.losses)

        # MTLR L2 regularizer
        if self.mtlr_head:
            l += self.l_mtlr * mtlr_sum_of_squares(self.t_layers[-1])
            if self.include_censoring_density:
                l += self.l_mtlr * mtlr_sum_of_squares(self.c_layers[-1])
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)            

            if self.dependent_censoring:

                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred[1]), axis=1) + self.tol

            else:
            
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll
    
    
    def mmd(self, x, s, beta=None):

        if beta is None:
            beta = get_median(tf.reduce_sum((x[:, tf.newaxis, :] - x[tf.newaxis, :, :]) ** 2, axis=-1))
        
        x0 = tf.boolean_mask(x, s == 0, axis=0)
        x1 = tf.boolean_mask(x, s == 1, axis=0)

        x0x0 = self._gaussian_kernel(x0, x0, beta)
        x0x1 = self._gaussian_kernel(x0, x1, beta)
        x1x1 = self._gaussian_kernel(x1, x1, beta)
        
        return tf.reduce_mean(x0x0) - 2. * tf.reduce_mean(x0x1) + tf.reduce_mean(x1x1)


    def _gaussian_kernel(self, x1, x2, beta=1.):
        return tf.exp(-1. * beta * tf.reduce_sum((x1[:, tf.newaxis, :] - x2[tf.newaxis, :, :]) ** 2, axis=-1))


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


class NMC(Model):
    
    def __init__(self,
                 e_layer_sizes=[256, 256], t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 n_bins=50,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NMC, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.n_bins = n_bins
        self.activation = activation
        self.tol = tol
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:
            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [
                    Sequential(cl)
                    for cl in self.c_layers
                ]
            else:
                self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
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
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    

    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)

            if self.dependent_censoring:
                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred[1]), axis=1) + self.tol
            else:
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


class NSurv(Model):
    
    def __init__(self,
                 t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 dependent_censoring=False,
                 n_bins=50,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NSurv, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.n_bins = n_bins
        self.activation=activation
        self.tol = tol
        
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        self.t_model = Sequential(self.t_layers)

        if self.include_censoring_density:
            if dependent_censoring:
                self.c_layers = [
                    [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
                    for i in range(2)
                ]
                self.c_model = [
                    Sequential(cl)
                    for cl in self.c_layers
                ]
            else:
                self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
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
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            t_pred, c_pred = self.forward_pass(x)

            if self.dependent_censoring:
                fc = tf.reduce_sum(yt * c_pred[0], axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred[1]), axis=1) + self.tol
            else:
                fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
                Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(Ft) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


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

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        e_pred = self.forward_pass(x)
        
        l1 = e_pred
        l2 = (1 - e_pred)
            
        ll = tf.cast(s, dtype=tf.float32) * tf.math.log(l1)
        ll += tf.cast((1 - s), dtype=tf.float32) * tf.math.log(l2)
        
        return -1 * ll


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
		tp = t_pred_cdf[idx != tti_idx, tti_idx]

		total += np.sum(tti_idx < tt_idx) # observed in i first
		concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

	return concordant / total


def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)


class MTLRDiffLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MTLRDiffLayer, self).__init__()

    def build(self, input_shape):
        self.a_shape = (input_shape[0], 1)

    def call(self, inputs):
        return tf.concat([tf.ones(self.a_shape), inputs], axis=-1) - tf.concat([inputs, tf.zeros(self.a_shape)], axis=-1)


def mtlr_sum_of_squares(layer):
    kernel = layer.get_weights()[0]
    return tf.reduce_sum((kernel[:, 1:] - kernel[:, :-1]) ** 2)


def get_batches(*arrs, batch_size=1):
    l = len(arrs[0])
    for ndx in range(0, l, batch_size):
        yield (arr[ndx:min(ndx + batch_size, l)] for arr in arrs)


def train_model(
    model, train_data, val_data, n_epochs,
    batch_size=50, learning_rate=1e-3, early_stopping_criterion=2,
    overwrite_output=True):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #@tf.function
    def train_step(x, y, s):
        with tf.GradientTape() as tape:
            train_loss, train_nll = model.loss(x, y, s)
            #print(train_loss, train_nll)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss, train_nll

    #@tf.function
    def test_step(x, y, s):
        val_loss, val_nll = model.loss(x, y, s)
        return val_loss, val_nll
    
    best_val_loss = np.inf
    no_decrease = 0

    for epoch_idx in range(n_epochs):

        train_losses = []
        train_nlls = []

        for batch_idx, (xt, yt, st) in enumerate(get_batches(*train_data, batch_size=batch_size)):

            train_loss, train_nll = train_step(xt, yt, st)

            train_losses.append(train_loss)
            train_nlls.append(train_nll)

        # Display metrics at the end of each epoch.
        #print('Epoch training loss: %.4f, NLL = %.4f' % (np.mean(batch_losses), np.mean(batch_nll)))

        val_losses = []
        val_nlls = []

        # Run a validation loop at the end of each epoch.
        for batch_idx, (xv, yv, sv) in enumerate(get_batches(*val_data, batch_size=batch_size)):

            val_loss, val_nll = test_step(xv, yv, sv)

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


def nll(t_true, t_pred, tol=1e-8):
    ncat = np.shape(t_pred)[1]
    nll_ = -1 * np.log(np.sum(onehot(t_true, ncategories=ncat) * t_pred, axis=1) + tol)
    return np.mean(nll_)


def evaluate_model(
    model, test_data, e_test,
    features=None,
    batch_size=50,
    dataset=None, random_state=None):
    
    modelname = type(model).__name__

    if random_state is None:
        random_state = np.nan
    
    test_losses = []
    test_nlls = []
    
    test_e_pred = []
    test_t_pred = []
    test_c_pred = []
    
    for batch_idx, (xt, yt, st) in enumerate(get_batches(*test_data, batch_size=batch_size)):
        
        test_loss, test_nll = model.loss(xt, yt, st)
        
        if modelname == 'NSurv':
            t_pred, c_pred = model(xt)
            test_t_pred.append(t_pred)
            test_c_pred.append(c_pred)
        elif modelname == 'MLP':
            e_pred = model(xt)
            test_e_pred.append(e_pred)
        else:
            e_pred, t_pred, c_pred = model(xt)
            test_e_pred.append(e_pred)
            test_t_pred.append(t_pred)
            test_c_pred.append(c_pred)
        
        test_losses.append(test_loss)
        test_nlls.append(test_nll)
    
    if modelname == 'NSurv':
        ttp = np.concatenate(test_t_pred, axis=0)
        num_bins = ttp.shape[1]
        e_auc = roc_auc_score(e_test, 1 - ttp @ np.arange(num_bins) / num_bins)
        true_rates, pred_rates = calibration_curve(e_test, 1 - ttp @ np.arange(num_bins) / num_bins, nbins=8)
        ci = discrete_ci(test_data[2], test_data[1], ttp)
    elif modelname == 'MLP':
        e_auc = roc_auc_score(e_test, np.concatenate(test_e_pred, axis=0))
        true_rates, pred_rates = calibration_curve(e_test, np.concatenate(test_e_pred, axis=0), nbins=8)
        ci = None
    else:
        e_auc = roc_auc_score(e_test, np.concatenate(test_e_pred, axis=0))
        true_rates, pred_rates = calibration_curve(e_test, np.concatenate(test_e_pred, axis=0), nbins=8)
        ci = discrete_ci(
            test_data[2], test_data[1],
            np.concatenate(test_e_pred, axis=0)[:, np.newaxis] * np.concatenate(test_t_pred, axis=0)
        )

    N_total = len(e_test)
    hl_stat, hl_pval = hosmer_lemeshow_cal_metric(true_rates, pred_rates, N_total)
    
    results = {
        'dataset': dataset,
        'random_state': random_state,
        'model': modelname,
        'ld': model.ld,
        'lr': model.lr,
        'avg_test_loss': np.mean(test_losses),
        'avg_test_nll': np.mean(test_nlls),
        'e_auc': e_auc,
        'y_ci': ci,
        'sample_N': N_total,
        'hl_stat': hl_stat,
        'hl_pval': hl_pval
    }

    for i, (tr, pr) in enumerate(zip(true_rates, pred_rates)):
        results['true_event_rate_quantile_%i' % i] = tr
        results['pred_event_rate_quantile_%i' % i] = pr

    if features is None:
        return results

    shared_features, tc_features, e_features = features

    results['N_shared'] = len(shared_features)
    results['N_distinct'] = len(tc_features)

    if modelname == 'NSurv':

        t_features = np.concatenate([tc_features, shared_features])

        results['t_in'], results['t_out'] = pull_split_avg(
            model.t_layers[0],
            np.concatenate([tc_features, shared_features]))
        results['t_in_max'], results['t_out_max'] = pull_split_max_avg(
            model.t_layers[0],
            np.concatenate([tc_features, shared_features]))

    elif modelname == 'MLP':

        results['e_in'], results['e_out'] = pull_split_avg(
            model.e_layers[0],
            np.concatenate([e_features, shared_features]))
        results['e_in_max'], results['e_out_max'] = pull_split_max_avg(
            model.e_layers[0],
            np.concatenate([e_features, shared_features]))

    elif modelname == 'NMC':

        results['e_in'], results['e_out'] = pull_split_avg(
            model.e_layers[0],
            np.concatenate([e_features, shared_features]))
        results['t_in'], results['t_out'] = pull_split_avg(
            model.t_layers[0],
            np.concatenate([tc_features, shared_features]))

        results['e_in_max'], results['e_out_max'] = pull_split_max_avg(
            model.e_layers[0],
            np.concatenate([e_features, shared_features]))
        results['t_in_max'], results['t_out_max'] = pull_split_max_avg(
            model.t_layers[0],
            np.concatenate([tc_features, shared_features]))

    else:

        if model.include_psi:
            results['e_in'], results['e_out'] = pull_split_avg(
                model.phi_layers[0],
                e_features)
            results['t_in'], results['t_out'] = pull_split_avg(
                model.omega_layers[0],
                tc_features)
            results['shared_in'], results['shared_out'] = pull_split_avg(
                model.psi_layers[0],
                shared_features)

            results['e_in_max'], results['e_out_max'] = pull_split_max_avg(
                model.phi_layers[0],
                e_features)
            results['t_in_max'], results['t_out_max'] = pull_split_max_avg(
                model.omega_layers[0],
                tc_features)
            results['shared_in_max'], results['shared_out_max'] = pull_split_max_avg(
                model.psi_layers[0],
                shared_features)

        else:
            results['e_in'], results['e_out'] = pull_split_avg(
                model.phi_layers[0],
                np.concatenate([e_features, shared_features]))
            results['t_in'], results['t_out'] = pull_split_avg(
                model.omega_layers[0],
                np.concatenate([tc_features, shared_features]))

            results['e_in_max'], results['e_out_max'] = pull_split_max_avg(
                model.phi_layers[0],
                np.concatenate([e_features, shared_features]))
            results['t_in_max'], results['t_out_max'] = pull_split_max_avg(
                model.omega_layers[0],
                np.concatenate([tc_features, shared_features]))
    
    return results


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


def calibration_curve(y_true, y_pred, nbins=10):
    
    bins = np.linspace(0, 1, nbins + 1)
    quantiles = np.quantile(y_pred, bins)

    avg_tr = []
    avg_pr = []

    for low, high in zip(quantiles[:-1], quantiles[1:]):
        idx = (y_pred >= low) & (y_pred <= high)
        avg_tr.append(np.mean(y_true[idx]))
        avg_pr.append(np.mean(y_pred[idx]))

    return np.array(avg_tr), np.array(avg_pr)


from scipy.stats import chi2

def hosmer_lemeshow_cal_metric(true, pred, N_total):
    N_per_group = N_total / len(true)
    stat = np.sum(((true - pred) ** 2) * N_per_group / (pred * (1 - pred)))
    return stat, chi2.logpdf(stat, len(true) - 2)

