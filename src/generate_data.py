import numpy as np

def onehot(arr, ncategories=None):
    if ncategories is None:
        ncategories = len(np.unique(arr))
    return np.eye(ncategories)[arr.astype(int)]

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

def generate_semi_synthetic(
    X, num_distinct_features, num_shared_features, num_bins, random_state,
    e_prob_spread=2.5, dependent_censoring=False, censoring_offset_in_sd=1.):
    
    n, m = X.shape
    
    rs = np.random.RandomState(random_state)

    features = rs.choice(np.arange(m), 2 * num_distinct_features + num_shared_features, replace=False)

    shared_features = features[:num_shared_features]
    tc_features = features[num_shared_features:-num_distinct_features]
    e_features = features[-num_distinct_features:]

    t_weights = rs.randn(num_distinct_features + num_shared_features)
    e_weights = rs.randn(num_distinct_features + num_shared_features)
    c_weights = rs.randn(num_distinct_features + num_shared_features)

    t = X[:, np.concatenate([shared_features, tc_features])] @ t_weights
    e_logits = X[:, np.concatenate([shared_features, e_features])] @ e_weights
    c = X[:, np.concatenate([shared_features, tc_features])] @ c_weights
    
    e_prob = sigmoid(e_prob_spread * (e_logits - e_logits.mean()) / e_logits.std())
    e = (rs.rand(len(e_prob)) < e_prob).astype(int)

    if dependent_censoring:
        c = c + e * censoring_offset_in_sd * np.std(c)
    
    y = e * np.minimum(c, t) + (1 - e) * c
    
    t_disc = np.zeros_like(t)
    c_disc = np.zeros_like(c)
    y_disc = np.zeros_like(y)
    
    for i in range(1, num_bins):
        
        prc = np.percentile(y, 100 * i / num_bins)
        
        t_disc = t_disc + (t > prc).astype(int)
        c_disc = c_disc + (c > prc).astype(int)
        y_disc = y_disc + (y > prc).astype(int)

    s = (y == t).astype(int)

    return {
        'y': y,
        'y_disc': y_disc,
        't': t,
        't_disc': t_disc,
        'c': c,
        'c_disc': c_disc,
        'e_prob': e_prob,
        'e': e,
        's': s,
        'shared_features': shared_features,
        'tc_features': tc_features,
        'e_features': e_features
    }


def generate_synth_censoring(
    X, y, s, num_e_features, num_bins, random_state, 
    e_prob_spread=2.5, dependent_censoring=False, censoring_offset_in_sd=1.):
    
    n, m = X.shape
    
    rs = np.random.RandomState(random_state)

    e_features = rs.choice(np.arange(m), num_e_features, replace=False)
    e_weights = rs.randn(num_e_features)

    e_logits = X[:, e_features] @ e_weights
    
    e_prob = sigmoid(e_prob_spread * (e_logits - e_logits.mean()) / e_logits.std())
    e = (rs.rand(len(e_prob)) < e_prob).astype(int)

    s = s * e # set to zero wherever e is 0

    if dependent_censoring:
        y = y + s * censoring_offset_in_sd * np.std(y)
    
    y_disc = np.zeros_like(y)
    
    for i in range(1, num_bins):
        prc = np.percentile(y, 100 * i / num_bins)
        y_disc = y_disc + (y > prc).astype(int)

    return {
        'y': y,
        'y_disc': y_disc,
        'e_prob': e_prob,
        'e': e,
        's': s,
        'shared_features': np.arange(m),
        'tc_features': np.arange(m),
        'e_features': e_features,
    }