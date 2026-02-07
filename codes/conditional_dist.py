import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming

"""
CONDITIONAL DISTRIBUTION P(TARGET|KEY)

Args:
 - data (pd.DataFrame): The original dataset.
 - syn_data (pd.DataFrame): Synthetic data generated from the original data. 
 - key (list[str] or str): Column(s) used as conditioning variables.
 - target (list[str] or str): Target column(s) containing sensitive information.
 - imputation (str, optional): Type of imputation method for unmatched keys. 
                               Options: [None, 'zero_risk', 'discard', 'naive', 'appr'].

Kwargs:
 - neighborhood (int): Number of neighbors to consider for "appr" imputation.

Returns:
 - cond_dist1 (pd.DataFrame): Conditional distribution of the original dataset.
 - cond_dist2 (pd.DataFrame): Conditional distribution of the synthetic dataset. 
"""

def compute_conditional_distributions(data, syn_data, key, target, imputation = None, **kwargs)
    nrow1 = len(data)
    concat_data = pd.concat([data, syn_data], axis = 0)

    key_cols = [key] if isinstance(key, str) else list(key)
    target_cols = [target] if isinstance(target, str) else list(target)

    key_tuples = pd.MultiIndex.from_frame(concat_data[key_cols]).to_list()
    target_tuples = pd.MultiIndex.from_frame(concat_data[target_cols]).to_list()

    k_codes, k_uniques = pd.factorize(pd.Index(key_tuples), sort=True)
    t_codes, t_uniques = pd.factorize(pd.Index(target_tuples), sort=True)

    nK, nT = len(k_uniques), len(t_uniques)

    counts1 = np.zeros((nK, nT), dtype=np.int64)
    counts2 = np.zeros((nK, nT), dtype=np.int64)
    result1 = np.zeros((nK, nT), dtype=np.float64)
    result2 = np.zeros((nK, nT), dtype=np.float64)
     
    np.add.at(counts1, (k_codes[:nrow1], t_codes[:nrow1]), 1)
    np.add.at(counts2, (k_codes[nrow1:], t_codes[nrow1:]), 1)

    row_sums1 = counts1.sum(axis=1, keepdims=True)
    row_sums2 = counts2.sum(axis=1, keepdims=True)

    unmatched_keys_idx = np.where(row_sums2.ravel() == 0)
    
    cond_prob1 = np.divide(counts1, row_sums1, out = result1, where=row_sums1 != 0)
    cond_prob2 = np.divide(counts2, row_sums2, out = result2, where=row_sums2 != 0)
    
    def join_or_str(x):
        return "_".join(map(str, x)) if isinstance(x, tuple) else str(x)

    comp_keys = [join_or_str(u) for u in k_uniques]
    unmatched_keys = [comp_keys[i] for i in unmatched_keys_idx[0]] 
    comp_targets = [join_or_str(u) for u in t_uniques]

    all_idx = pd.MultiIndex.from_product([comp_keys, comp_targets], names=["composite_key", "composite_target"])
    
    
    cond_dist1 = pd.DataFrame({"count": counts1.ravel(), "cond_prob": cond_prob1.ravel()}, index=all_idx).reset_index()
    
    if imputation is not None:
        cond_dist2 = pd.DataFrame({"count": counts2.ravel(), "cond_prob": cond_prob2.ravel(), "imputed_prob": cond_prob2.ravel()}, index=all_idx).reset_index()
        if imputation == "naive":
            for k in unmatched_keys:
                impute_indices = np.where(cond_dist2["composite_key"]==k)[0]
                
                total_count = cond_dist2["count"].sum()
                target_sums = cond_dist2.groupby("composite_target")["count"].sum()
                naive_probs = target_sums / total_count

                cond_dist2.loc[impute_indices, "imputed_prob"] = cond_dist2.loc[impute_indices, 'composite_target'].map(naive_probs)
                     
            return cond_dist1, cond_dist2

        elif imputation == "appr":
            neighborhood = kwargs.get('neighborhood')
            
            if neighborhood is None:
                neighborhood = 1
                warnings.warn("For 'appr imputation', a 'neighborhood' value was not provided. "
                              "Using default value of 1.", 
                              stacklevel = 2)
                
            for k in unmatched_keys:
                impute_indices = np.where(cond_dist2["composite_key"]==k)[0]

                total_count = cond_dist2["count"].sum()
                key_sums = cond_dist2.groupby("composite_key")["count"].sum()
                probs = key_sums / total_count

                cond_dist2['distance'] = cond_dist2['composite_key'].apply(
                    lambda x: hamming(x.split("_"), k.split("_")) * len(k.split("_")))
                neighbor_dist = cond_dist2[(0 < cond_dist2.distance) & (cond_dist2.distance <= neighborhood)]
                neighbor_keys = np.unique(neighbor_dist.composite_key)
                multiplier = probs[neighbor_keys] / np.sum(probs[neighbor_keys])

                appr_probs = neighbor_dist.groupby("composite_target")["cond_prob"].apply(lambda x: np.sum(x * multiplier.to_numpy()))
                
                cond_dist2.loc[impute_indices, 'imputed_prob'] = cond_dist2.loc[impute_indices, 'composite_target'].map(appr_probs)

                cond_dist2.drop(["distance"], axis = 1, inplace = True)

            return cond_dist1, cond_dist2
        
        else:
            cond_dist2 = pd.DataFrame({"count": counts2.ravel(), "cond_prob": cond_prob2.ravel()}, index=all_idx).reset_index()
            return cond_dist1, cond_dist2

    else:
        cond_dist2 = pd.DataFrame({"count": counts2.ravel(), "cond_prob": cond_prob2.ravel()}, index=all_idx).reset_index()
        return cond_dist1, cond_dist2
