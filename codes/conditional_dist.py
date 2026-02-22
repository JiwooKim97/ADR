import warnings
import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cdist


def compute_conditional_distributions(data, syn_data, key, target, imputation = None, **kwargs):
    
    """
    Computes the conditional distribution of the target variables given key variables, 
    denoted as $P(\text{TARGET} \mid \text{KEY})$, for both original and synthetic datasets.
    
    Args:
     - data (pd.DataFrame): The original dataset containing sensitive information.
     - syn_data (pd.DataFrame): Synthetic data generated from the original data. 
     - key (list[str] or str): Column(s) used as conditioning variables (quasi-identifiers).
     - target (list[str] or str): Target column(s) containing sensitive information.
     - imputation (str, optional): Method for handling unmatched keys. 
                                   Options: [None, 'zero_risk', 'discard', 'naive', 'appr'].
    
    Kwargs:
     - neighborhood (int): Number of neighbors to consider for "appr" imputation (Default: 1).
    
    Returns:
     - cond_dist1 (pd.DataFrame): Conditional distribution of the original dataset.
     - cond_dist2 (pd.DataFrame): Conditional distribution of the synthetic dataset. 
    """
    
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
    cond_dist2 = pd.DataFrame({"count": counts2.ravel(), "cond_prob": cond_prob2.ravel()}, index=all_idx).reset_index()

    if imputation in ["naive", "appr"]:
        cond_dist2["imputed_prob"] = cond_dist2["cond_prob"]
                
        if imputation == "naive":
            total_count = cond_dist2["count"].sum()
            target_sums = cond_dist2.groupby("composite_target")["count"].transform("sum")
            naive_probs = target_sums / total_count

            mask = cond_dist2["composite_key"].isin(unmatched_keys)
            cond_dist2.loc[mask, "imputed_prob"] = naive_probs[mask]         

        elif imputation == "appr":
            neighborhood = kwargs.get('neighborhood')
            
            if neighborhood is None:
                neighborhood = 1
                warnings.warn("For 'appr imputation', a 'neighborhood' value was not provided. "
                              "Using default value of 1.", 
                              stacklevel = 2)
            
            total_count = cond_dist2["count"].sum()
            probs = cond_dist2.groupby("composite_key")["count"].sum() / total_count

            
            unique_keys = cond_dist2["composite_key"].unique()
            split_keys = np.array([k.split('_') for k in unique_keys])

            encoded_keys = np.zeros(split_keys.shape, dtype=int)
            for col in range(split_keys.shape[1]):
                encoded_keys[:, col] = pd.factorize(split_keys[:, col])[0]
            
            dist_matrix = cdist(encoded_keys, encoded_keys, metric='hamming') * split_keys.shape[1]

            key_to_idx = {key: i for i, key in enumerate(unique_keys)}

            imputation_map = {}

            for k in unmatched_keys:
                if k not in key_to_idx: 
                    continue   

                k_idx = key_to_idx[k]
                dists = dist_matrix[k_idx]
                
                neighbor_mask = (dists > 0) & (dists <= neighborhood)
                neighbor_keys = unique_keys[neighbor_mask]
                
                if len(neighbor_keys) == 0: 
                    continue

                p_neighbors = probs[neighbor_keys]
                multiplier = p_neighbors / p_neighbors.sum()
                
                neighbor_data = cond_dist2[cond_dist2["composite_key"].isin(neighbor_keys)].copy()
                neighbor_data['weight'] = neighbor_data['composite_key'].map(multiplier)
                
                neighbor_data['weighted_prob'] = neighbor_data['cond_prob'] * neighbor_data['weight']
                appr_probs = neighbor_data.groupby("composite_target")['weighted_prob'].sum()
                
                imputation_map[k] = appr_probs
                
            for k, appr_probs in imputation_map.items():
                mask = cond_dist2["composite_key"] == k
                cond_dist2.loc[mask, 'imputed_prob'] = cond_dist2.loc[mask, 'composite_target'].map(appr_probs)

    return cond_dist1, cond_dist2
                

                
