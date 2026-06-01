import risk_weight_fns as rwf
import pandas as pd
import numpy as np
from conditional_dist import compute_conditional_distributions


class ADR:
    def __init__(self, 
                 data, 
                 syn_data, 
                 key, 
                 target, 
                 risk = "inner_product_similarity", 
                 weight = "OD_prevalence",
                 imputation="constant_risk",
                 use_deterministic = False,
                 normalize=True, 
                 show_all=False, 
                 **kwargs):
        """
        Initializes a new instance of the ADR class.
        
        Args:
         - data (pd.DataFrame): The original dataset containing sensitive information.
         - syn_data (pd.DataFrame): Synthetic data generated from the original data.
         - key (list[str] or str): Column(s) used as conditioning variables (quasi-identifiers).
         - target (list[str] or str): Target column(s) containing sensitive information.
         - risk (str): Risk function used for measuring ADR.
         - weight (str): Weight function used for aggregating individual key risks.
         - imputation (str, optional): Method for handling unmatched keys. 
                                       Options: ['constant_risk', 'exclusion', 'marginal', 'neighborhood_appr'].
         - use_deterministic (bool, optional): 
         - normalize (bool, optional): Whether to normalize weight functions.
         - show_all (bool, optional): Whether to return individual key-level results.
        
        Kwargs:
         - neighborhood (int): Number of neighbors to consider for 'appr' imputation (Default: 1).
         - alpha (float): A positive value required for calculating 'KL_similarity' and 'wasserstein_similarity' risk functions (Default: 1.0).
         - positive_target_value: The specific target value required for 'precision' and 'recall'  risk functions, as well as the 'prediction_positive' weight function. (Default: the least frequent target among all composite targets).
         - constant_risk_value (float): The risk value to assign for original-only keys when imputation='constant_risk'. Should be >= 0. (Default: 0.0).
        """
        
        self.data = data
        self.syn_data = syn_data
        self.key = key
        self.target = target
        self.risk = risk
        self.weight = weight
        self.imputation = imputation
        self.use_deterministic = use_deterministic
        self.normalize = normalize
        self.show_all = show_all
        self.configs = kwargs

        self.orig_cond_dist = None
        self.syn_cond_dist = None

    def _precompute_key_target_uniques(self):
        key_cols = [self.key] if isinstance(self.key, str) else list(self.key)
        target_cols = [self.target] if isinstance(self.target, str) else list(self.target)
        
        all_keys_df = pd.concat([self.data[key_cols], self.syn_data[key_cols]], axis=0)
        key_tuples = pd.MultiIndex.from_frame(all_keys_df).to_list()
        _, self.k_uniques = pd.factorize(pd.Index(key_tuples), sort=True)
        
        all_targets_df = pd.concat([self.data[target_cols], self.syn_data[target_cols]], axis=0)
        target_tuples = pd.MultiIndex.from_frame(all_targets_df).to_list()
        _, self.t_uniques = pd.factorize(pd.Index(target_tuples), sort=True)

    
    def _precompute_global_stats(self):
        prob_col = 'imputed_prob' if 'imputed_prob' in self.syn_cond_dist.columns else 'cond_prob'
        
        self.p1_matrix_df = self.orig_cond_dist.pivot(index='composite_key', columns='composite_target', values='cond_prob').fillna(0)
        self.p2_matrix_df = self.syn_cond_dist.pivot(index='composite_key', columns='composite_target', values=prob_col).fillna(0)
        
        self.p1_matrix = self.p1_matrix_df.values
        self.p2_matrix = self.p2_matrix_df.values

        self.target_dist = self.orig_cond_dist.groupby('composite_target')['count'].sum() / self.orig_cond_dist['count'].sum()
        self.target_vector = self.target_dist.reindex(self.p1_matrix_df.columns).fillna(0).values
        self.target_matrix = np.tile(self.target_vector, (self.p1_matrix.shape[0], 1))
        
        self.target_array = np.unique(self.orig_cond_dist['composite_target'])
        self.target_to_idx = {name: i for i, name in enumerate(self.target_array)}
        
        self.mode1_vector = self.target_array[np.argmax(self.p1_matrix, axis=1)]
        self.mode2_vector = self.target_array[np.argmax(self.p2_matrix, axis=1)]
        
        self.deterministic_keys2 = self.syn_cond_dist.loc[self.syn_cond_dist[prob_col] == 1.0, 'composite_key'].unique().tolist()
        self.best_targets_df = pd.DataFrame({
            'composite_key': self.p2_matrix_df.index, 
            'composite_target': self.mode2_vector
        })

    def prepare_data(self, data1 = None, data2 = None):
        current_data1 = data1 if data1 is not None else self.data
        current_data2 = data2 if data2 is not None else self.syn_data
        
        self._precompute_key_target_uniques()
        
        actual_imputation = self.imputation
        if (current_data2 is self.data) or (current_data2 is current_data1):
            actual_imputation = None
            
        self.orig_cond_dist, self.syn_cond_dist = compute_conditional_distributions(current_data1, 
                                                                                    current_data2, 
                                                                                    self.key, 
                                                                                    self.target, 
                                                                                    self.k_uniques,
                                                                                    self.t_uniques,
                                                                                    actual_imputation, 
                                                                                    **self.configs)
        
        
        self._precompute_global_stats()


        orig_key_counts = self.orig_cond_dist.groupby("composite_key")["count"].sum()
        syn_key_counts = self.syn_cond_dist.groupby("composite_key")["count"].sum()

        self.all_keys = self.p1_matrix_df.index.values
        self.is_intersection = (orig_key_counts.reindex(self.all_keys, fill_value=0) > 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) > 0)
        self.is_only_orig = (orig_key_counts.reindex(self.all_keys, fill_value=0) > 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) == 0)
        self.is_only_syn = (orig_key_counts.reindex(self.all_keys, fill_value=0) == 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) > 0)
        

    
    def calculate(self, data1 = None, data2 = None, risk = None, weight = None, marginal_reference = False):
        
        """
        Calculates the Attribute Disclosure Risk (ADR) score between two datasets.

        Args:
         - data1 (pd.DataFrame, optional): The reference dataset used as the baseline for risk assessment (e.g., the original data).
         - data2 (pd.DataFrame, optional): The evaluation dataset to be assessed against the reference data (e.g., the synthetic data).
         - risk (str, optional): The specific risk function to be applied for the calculation.
         - weight (str, optional): The weight function used to compute the final aggregated ADR score.
         - marginal_reference (bool, optional): Whether to use the marginal distribution of the original target variable, instead of the conditional distribution.

        Returns: 
         - total_adr (float): The final aggregated Attribute Disclosure Risk value.
         - score_df (pd.DataFrame, optional): Detailed results for each key.
        
        """
        
        current_data1 = data1 if data1 is not None else self.data
        current_data2 = data2 if data2 is not None else self.syn_data

        name1 = "original data"
        name2 = "original data" if data2 is not None else "synthetic data"
        
        self.prepare_data(current_data1, current_data2)
        
        current_risk = risk if risk else self.risk
        current_weight = weight if weight else self.weight
        
        risk_func = getattr(rwf, current_risk)
        weight_func = getattr(rwf, current_weight)

        self.disclosive_keys = self.deterministic_keys2 if self.use_deterministic else None

        params = {
            "p1": self.p1_matrix,
            "p2": self.p2_matrix if not marginal_reference else self.target_matrix,
            "mode1": self.mode1_vector,
            "mode2": self.mode2_vector,
            "target_to_idx": self.target_to_idx,
            "cond_dist1": self.orig_cond_dist,
            "cond_dist2": self.syn_cond_dist,
            "deterministic_keys2": self.disclosive_keys,
            "best_targets_df": self.best_targets_df,
            "normalize": self.normalize,
            **self.configs
        }

        risk_vector = risk_func(**params)
        weight_vector = weight_func(**params)


        # 1. Only Synthetic Keys
        risk_vector[self.is_only_syn] = 0.0

        # 2. Original-only Keys
        if self.imputation == "exclusion":
            weight_vector[~self.is_intersection] = 0.0
            w_sum = np.sum(weight_vector)
            if w_sum > 0:
                weight_vector /= w_sum

        elif self.imputation == "constant_risk":
            const_val = self.configs.get("constant_risk_value", 0.0)
            
            if const_val < 0:
                raise ValueError("constant_risk_value must be a number greater than or equal to 0.")
                
            risk_vector[self.is_only_orig] = const_val
        
        
        # 3. ADR Score = Sum(Risk * Weight)
        final_scores = risk_vector * weight_vector
        
        total_adr = np.sum(final_scores)


        if self.show_all:
            score_df = pd.DataFrame({"key": self.all_keys, "risk": risk_vector, "weight": weight_vector, "score": final_scores}).sort_values("key").reset_index(drop=True)
            return total_adr, score_df

        return total_adr
