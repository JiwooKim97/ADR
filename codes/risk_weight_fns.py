import warnings
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as w_dist

################################################################################################
## Risk Functions (similarity-based / prediction-accuracy-based / taks-oriented )
################################################################################################

## Similarity-based risk functions

def inner_product_similarity(p1, p2, **kwargs):
    return np.sum(p1*p2)


def cosine_similarity(p1, p2, **kwargs):
    numerator = np.sum(p1*p2)
    denominator = np.sqrt(np.sum((p1)**2)) * np.sqrt(np.sum((p2)**2))

    if denominator == 0.0:
        return 0.0
    else:
        return (numerator / denominator)


def bhattacharyya_coefficient(p1, p2, **kwargs):
    sum_sqrt = np.sum(np.sqrt(p1*p2))
    return sum_sqrt


def total_variation(p1, p2, **kwargs):
    sum_abs = np.sum(np.abs(p1-p2))
    return 1-(1/2*sum_abs)


def hellinger_distance(p1, p2, **kwargs):
    sqrt_term = np.sqrt(1/2*np.sum((np.sqrt(p1)-np.sqrt(p2))**2))
    return 1-sqrt_term


def KL_divergence(p1, p2, **kwargs):
    alpha = kwargs.get('alpha')
    
    if alpha is None:
        alpha = 1
        warnings.warn("For 'KL_divergence', a 'alpha' value was not provided. " 
                      "Using default value of 1.", 
                      stacklevel = 2)
        
    mask = (p1 > 0) & (p2 > 0)
    KL = np.sum(p1[mask] * np.log(p1[mask] / p2[mask]))
    return np.exp(-alpha * KL)


def JS_divergence(p1, p2, **kwargs):
    p3 = (p1+p2)/2
    mask_1 = (p1 > 0) & (p3 > 0)
    mask_2 = (p2 > 0) & (p3 > 0)
    KL_1 = np.sum(p1[mask_1]*np.log(p1[mask_1]/p3[mask_1]))
    KL_2 = np.sum(p2[mask_2]*np.log(p2[mask_2]/p3[mask_2]))
    JS = 1/2*(KL_1+KL_2)
    return 1-(JS/np.log(2))


def wasserstein_distance(p1, p2, **kwargs):
    alpha = kwargs.get('alpha')
    
    if alpha is None:
        alpha = 1
        warnings.warn("For 'wasserstein_distance', a 'alpha' value was not provided. "
                      "Using default value of 1.", 
                      stacklevel = 2)
        
    return np.exp(-alpha*w_dist(p1, p2))


## prediction-accuracy-based risk functions

def accuracy(key_df1, mode2, **kwargs):
    return key_df1.loc[key_df1["composite_target"] == mode2, "cond_prob"].item()


def tcap_similarity(p1, p2, **kwargs):
    return np.sum(p1[p2 == 1])


def mode_similarity(mode1, mode2, **kwargs):
    return 1.0 if mode1 == mode2 else 0.0


## task-oriented risk functions

def precision(cond_dist1, key_df1, mode2, **kwargs):
    positive_target_value = kwargs.get('positive_target_value')

    if positive_target_value is None:
        # define the positive target as the least frequent among all composite targets
        positive_target_value = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'precision', a 'positive_target_value' was not provided. "
                      "Set the positive target as the least frequent among all composite targets.", 
                      stacklevel = 2)
    
    return key_df1.loc[key_df1["composite_target"] == positive_target_value, "cond_prob"].item() if mode2 == positive_target_value else 0.0


def recall(cond_dist1, key_df1, mode2, **kwargs):
    positive_target_value = kwargs.get('positive_target_value')

    if positive_target_value is None:
        # define the positive target as the least frequent among all composite targets
        positive_target_value = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'recall', a 'positive_target_value' was not provided. "
                      "Set the positive target as the least frequent among all composite targets.", 
                      stacklevel = 2)
    
    numerator = key_df1.loc[key_df1["composite_target"] == positive_target_value, "cond_prob"].item() if mode2 == positive_target_value else 0.0
    denominator = np.sum(cond_dist1.loc[cond_dist1["composite_target"] == positive_target_value, "count"]) / np.sum(cond_dist1["count"])

    if denominator == 0:
        return 0.0
    else:
        return (numerator / denominator)



################################################################################################
## Weight Functions (prevalence-based / concentration-based)
################################################################################################

## prevalence-based weights

def weight_key_proportion1(cond_dist1, key_df1, **kwargs):
    total_counts = np.sum(cond_dist1["count"])
    key_counts = np.sum(key_df1["count"])

    if total_counts == 0:
        return 0.0
    else:
        return key_counts / total_counts
    
    
def weight_key_proportion2(cond_dist2, key_df2, c = None, **kwargs):   
    total_counts = np.sum(cond_dist2["count"])
    key_counts = np.sum(key_df2["count"])
    
    if total_counts == 0:
        return 0.0
    else:
        return key_counts / total_counts


def weight_tcap(key, cond_dist2, key_df2, deterministic_keys2, **kwargs):
    if not deterministic_keys2:
        return 0.0
    else:
        filtered_df = cond_dist2[cond_dist2["composite_key"].isin(deterministic_keys2)]
        total_counts = np.sum(filtered_df["count"])
        key_counts = np.sum(key_df2["count"]) if key in deterministic_keys2 else 0

        if total_counts == 0:
            return 0.0
        else:
            return key_counts / total_counts

    
def weight_precision(key, cond_dist1, key_df1, best_targets_df, **kwargs):
    positive_target_value = kwargs.get('positive_target_value')

    if positive_target_value is None:
        # define the positive target as the least frequent among all composite targets
        positive_target_value = cond_dist1.groupby("composite_target")["count"].sum().idxmin()
        warnings.warn("For 'precision', a 'positive_target_value' was not provided. "
                      "Set the positive target as the least frequent among all composite targets.", 
                      stacklevel = 2)
        
    positive_keys = best_targets_df.loc[best_targets_df['composite_target'] == positive_target_value, 'composite_key'].tolist()

    if not positive_keys:
        return 0.0
    else:
        filtered_df = cond_dist1[cond_dist1["composite_key"].isin(positive_keys)]
        total_counts = np.sum(filtered_df["count"])
        key_counts = np.sum(key_df1["count"]) if key in positive_keys else 0

        if total_counts == 0:
            return 0.0
        else:
            return key_counts / total_counts


## concentration-based weights

def negentropy(cond_dist1, p1, normalize, **kwargs):
    if not np.isclose(np.sum(p1), 1, atol = 1e-9):
        raise ValueError("The sum of probabilities must be 1.")

    mask = p1 > 0
    H_x = -np.sum(p1[mask] * np.log2(p1[mask]))

    k = len(p1)
    H_uniform = np.log2(k)

    if normalize:
        mask_ = cond_dist1["cond_prob"] > 0
        total_weights = (
            len(np.unique(cond_dist1["composite_key"]))*np.log2(k) 
            + np.sum(cond_dist1["cond_prob"][mask_] * np.log2(cond_dist1["cond_prob"][mask_])))

    return (H_uniform - H_x) / total_weights if normalize else H_uniform - H_x
    

def gini_impurity(cond_dist1, p1, normalize, **kwargs):
    if normalize:
        total_weights = np.sum(cond_dist1["cond_prob"]**2)

    return (np.sum(p1**2) / total_weights) if normalize else np.sum(p1**2)
