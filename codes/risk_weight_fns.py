import warnings
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as w_dist


# -------------------------------------------------------------------------------------------------
# Similarity-Based Risk Functions
# -------------------------------------------------------------------------------------------------
def inner_product_similarity(p1, p2, **kwargs):
    return np.sum(p1 * p2, axis=1)


def cosine_similarity(p1, p2, **kwargs):
    numerator = np.sum(p1 * p2, axis=1)
    denom = np.sqrt(np.sum(p1**2, axis=1)) * np.sqrt(np.sum(p2**2, axis=1))
    return np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0)


def bhattacharyya_coefficient(p1, p2, **kwargs):
    return np.sum(np.sqrt(p1 * p2), axis=1)


def total_variation(p1, p2, **kwargs):
    return 1 - (0.5 * np.sum(np.abs(p1 - p2), axis=1))


def hellinger_distance(p1, p2, **kwargs):
    sqrt_term = np.sqrt(0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2))**2, axis=1))
    return 1 - sqrt_term

def KL_divergence(p1, p2, **kwargs):
    alpha = kwargs.get('alpha')
    
    if alpha is None:
        alpha = 1
        warnings.warn("For 'KL_divergence', a 'alpha' value was not provided. " 
                      "Using default value of 1.", 
                      stacklevel = 2)
        
    eps = 1e-10
    kl = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)), axis=1)
    return np.exp(-alpha * kl)

def JS_divergence(p1, p2, **kwargs):
    p3 = (p1 + p2) / 2

    eps = 1e-10
    kl_1 = np.sum(p1 * np.log((p1 + eps) / (p3 + eps)), axis=1)
    kl_2 = np.sum(p2 * np.log((p2 + eps) / (p3 + eps)), axis=1)
    
    js = 0.5 * (kl_1 + kl_2)
    return 1 - (js / np.log(2))

def wasserstein_distance(p1, p2, **kwargs):
    alpha = kwargs.get('alpha')
    if alpha is None:
        alpha = 1
        warnings.warn("For 'wasserstein_distance', a 'alpha' value was not provided. "
                      "Using default value of 1.", stacklevel=2)
    
    n_keys = p1.shape[0]
    w_distances = np.zeros(n_keys)
    target_values = np.arange(p1.shape[1])

    sum1 = np.sum(p1, axis=1)
    sum2 = np.sum(p2, axis=1)
    
    valid_indices = np.where((sum1 > 0) & (sum2 > 0))[0]
    
    for i in valid_indices:
        w_distances[i] = w_dist(target_values, target_values, p1[i], p2[i])
        
    return np.exp(-alpha * w_distances)


# -------------------------------------------------------------------------------------------------
# Prediction-Accuracy-Based Risk Functions
# -------------------------------------------------------------------------------------------------
def accuracy(p1, mode2, target_to_idx, **kwargs):
    n_keys = p1.shape[0]
    row_indices = np.arange(n_keys)
    col_indices = np.array([target_to_idx[m] for m in mode2])
    return p1[row_indices, col_indices]


def tcap_similarity(p1, p2, **kwargs):
    mask = (p2 == 1.0)
    return np.sum(p1 * mask, axis=1)


def mode_similarity(mode1, mode2, **kwargs):
    return (mode1 == mode2).astype(float)


# -------------------------------------------------------------------------------------------------
# Task-Oriented Risk Functions
# -------------------------------------------------------------------------------------------------
def precision(p1, mode2, target_to_idx, cond_dist1, **kwargs):
    pos_target = kwargs.get('positive_target_value')

    if pos_target is None:
        pos_target = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'precision', a 'positive_target_value' was not provided. "
                      "Set the positive target as the least frequent among all composite targets.", 
                      stacklevel=2)

    pos_idx = target_to_idx[pos_target]
    p1_pos_probs = p1[:, pos_idx]
    
    mask = (mode2 == pos_target).astype(float)
    return p1_pos_probs * mask


def recall(p1, mode2, target_to_idx, cond_dist1, **kwargs):
    pos_target = kwargs.get('positive_target_value')

    if pos_target is None:
        pos_target = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'recall', a 'positive_target_value' was not provided. "
                      "Set the positive target as the least frequent among all composite targets.", 
                      stacklevel=2)

    pos_idx = target_to_idx[pos_target]
    numerator = p1[:, pos_idx] * (mode2 == pos_target).astype(float)
    

    total_count = cond_dist1["count"].sum()
    pos_target_count = cond_dist1.loc[cond_dist1["composite_target"] == pos_target, "count"].sum()
    
    denominator = pos_target_count / total_count

    if denominator == 0:
        return np.zeros_like(numerator)
    else:
        return numerator / denominator


# -------------------------------------------------------------------------------------------------
# Prevalence-Based Weights
# -------------------------------------------------------------------------------------------------
def weight_key_proportion1(cond_dist1, **kwargs):
    key_counts = cond_dist1.groupby("composite_key")["count"].sum().values
    total_counts = np.sum(key_counts)

    if total_counts == 0:
        return np.zeros_like(key_counts, dtype=float)
    
    return key_counts / total_counts


def weight_key_proportion2(cond_dist2, **kwargs):
    key_counts = cond_dist2.groupby("composite_key")["count"].sum().values
    total_counts = np.sum(key_counts)

    if total_counts == 0:
        return np.zeros_like(key_counts, dtype=float)
    
    return key_counts / total_counts


def weight_tcap(cond_dist2, deterministic_keys2, **kwargs):
    key_stats = cond_dist2.groupby("composite_key")["count"].sum()
    
    if not deterministic_keys2:
        return np.zeros(len(key_stats), dtype=float)

    det_counts = key_stats.loc[deterministic_keys2]
    total_det_counts = det_counts.sum()

    if total_det_counts == 0:
        return np.zeros(len(key_stats), dtype=float)

    weights = (det_counts / total_det_counts).reindex(key_stats.index, fill_value=0.0)
    
    return weights.values


def weight_precision(cond_dist1, best_targets_df, **kwargs):
    pos_target = kwargs.get('positive_target_value')

    if pos_target is None:
        pos_target = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'weight_precision', a 'positive_target_value' was not provided.", stacklevel=2)

    
    positive_keys = best_targets_df.loc[best_targets_df['composite_target'] == pos_target, 'composite_key'].unique()
    key_stats = cond_dist1.groupby("composite_key")["count"].sum()

    if len(positive_keys) == 0:
        return np.zeros(len(key_stats), dtype=float)

    pos_key_counts = key_stats.loc[positive_keys]
    total_pos_counts = pos_key_counts.sum()

    if total_pos_counts == 0:
        return np.zeros(len(key_stats), dtype=float)

    weights = (pos_key_counts / total_pos_counts).reindex(key_stats.index, fill_value=0.0)

    return weights.values
    

# -------------------------------------------------------------------------------------------------
# Concentration-Based Weights
# -------------------------------------------------------------------------------------------------
def negentropy(p1, normalize, **kwargs):
    mask = p1 > 0
    p1_safe = np.where(mask, p1, 1.0)
    h_matrix = -np.sum(np.where(mask, p1 * np.log2(p1_safe), 0.0), axis=1)

    k = p1.shape[1]
    h_uniform = np.log2(k)

    negentropy_vector = h_uniform - h_matrix

    if normalize:
        total_weights = np.sum(negentropy_vector)
        
        if total_weights == 0:
            return np.zeros_like(negentropy_vector)
        
        return negentropy_vector / total_weights
    
    return negentropy_vector


def gini_impurity(p1, normalize, **kwargs):
    gini_vector = np.sum(p1**2, axis=1)

    if normalize:
        total_weights = np.sum(p1**2)
        
        if total_weights == 0:
            return np.zeros_like(gini_vector)
        
        return gini_vector / total_weights

    return gini_vector
