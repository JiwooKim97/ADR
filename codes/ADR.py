import risk_weight_fns as rwf
import pandas as pd
import numpy as np
import time
from conditional_dist import compute_conditional_distributions


class ADR:
    def __init__(self, 
                 data, 
                 syn_data, 
                 key, 
                 target, 
                 risk = "inner_product_similarity", 
                 weight = "weight_key_proportion1",
                 imputation=None, 
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
                                       Options: [None, 'zero_risk', 'discard', 'naive', 'appr'].
         - normalize (bool, optional): Whether to normalize concentration-based weight functions.
         - show_all (bool, optional): Whether to return individual key-level results.
        
        Kwargs:
         - neighborhood (int): Number of neighbors to consider for 'appr' imputation (Default: 1).
         - alpha (float): A positive value required for calculating 'KL_divergence' and 'wasserstein_distance' risk functions (Default: 1.0).
         - positive_target_value: The specific target value required for 'precision' and 'recall'  risk functions, as well as the 'weight_precision' weight function. 
                                  (Default: the least frequent target among all composite targets).
        """
        
        self.data = data
        self.syn_data = syn_data
        self.key = key
        self.target = target
        self.risk = risk
        self.weight = weight
        self.imputation = imputation
        self.normalize = normalize
        self.show_all = show_all
        self.configs = kwargs

        self.orig_cond_dist = None
        self.syn_cond_dist = None

    
    def _precompute_global_stats(self):
        prob_col = 'imputed_prob' if self.imputation in ["naive", "appr"] else 'cond_prob'
        
        self.p1_matrix_df = self.orig_cond_dist.pivot(index='composite_key', columns='composite_target', values='cond_prob').fillna(0)
        self.p2_matrix_df = self.syn_cond_dist.pivot(index='composite_key', columns='composite_target', values=prob_col).fillna(0)
        
        self.p1_matrix = self.p1_matrix_df.values
        self.p2_matrix = self.p2_matrix_df.values
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
        
        cond_start = time.time()

        self.orig_cond_dist, self.syn_cond_dist = compute_conditional_distributions(current_data1, 
                                                                                    current_data2, 
                                                                                    self.key, 
                                                                                    self.target, 
                                                                                    self.imputation, 
                                                                                    **self.configs)
        
        cond_end = time.time()
        self.cond_time = cond_end - cond_start
        print(f"Conditional distributions computed successfully! (Elapsed time: {self.cond_time:.2f}s)")
        print("-"*80)
        
        self._precompute_global_stats()


        orig_key_counts = self.orig_cond_dist.groupby("composite_key")["count"].sum()
        syn_key_counts = self.syn_cond_dist.groupby("composite_key")["count"].sum()

        self.all_keys = self.p1_matrix_df.index.values
        self.is_intersection = (orig_key_counts.reindex(self.all_keys, fill_value=0) > 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) > 0)
        self.is_only_orig = (orig_key_counts.reindex(self.all_keys, fill_value=0) > 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) == 0)
        self.is_only_syn = (orig_key_counts.reindex(self.all_keys, fill_value=0) == 0) & (syn_key_counts.reindex(self.all_keys, fill_value=0) > 0)
        

    
    def calculate(self, data1 = None, data2 = None, risk = None, weight = None):
        
        """
        Calculates the Attribute Disclosure Risk (ADR) score between two datasets.

        Args:
         - data1 (pd.DataFrame, optional): The reference dataset used as the baseline for risk assessment (e.g., the original data).
         - data2 (pd.DataFrame, optional): The evaluation dataset to be assessed against the reference data (e.g., the synthetic data).
         - risk (str, optional): The specific risk function to be applied for the calculation.
         - weight (str, optional): The weight function used to compute the final aggregated ADR score.

        Returns: 
         - total_adr (float): The final aggregated Attribute Disclosure Risk value.
         - score_df (pd.DataFrame, optional): Detailed results for each key.    
        """
        
        current_data1 = data1 if data1 is not None else self.data
        current_data2 = data2 if data2 is not None else self.syn_data

        name1 = "original data"
        name2 = "original data" if data2 is not None else "synthetic data"
        
        print(f"Computing conditional distributions for {name1} and {name2}...")
        self.prepare_data(current_data1, current_data2)

        print(f"Calculating ADR between {name1} and {name2}...")
        start = time.time()
        
        current_risk = risk if risk else self.risk
        current_weight = weight if weight else self.weight
        
        risk_func = getattr(rwf, current_risk)
        weight_func = getattr(rwf, current_weight)

        params = {
            "p1": self.p1_matrix,
            "p2": self.p2_matrix,
            "mode1": self.mode1_vector,
            "mode2": self.mode2_vector,
            "target_to_idx": self.target_to_idx,
            "cond_dist1": self.orig_cond_dist,
            "cond_dist2": self.syn_cond_dist,
            "deterministic_keys2": self.deterministic_keys2,
            "best_targets_df": self.best_targets_df,
            "normalize": self.normalize,
            **self.configs
        }

        risk_vector = risk_func(**params)
        weight_vector = weight_func(**params)

        # 1. Only Synthetic Keys
        risk_vector[self.is_only_syn] = 0.0

        # 2. Original-only Keys
        if self.imputation == "discard":
            weight_vector[~self.is_intersection] = 0.0
            w_sum = np.sum(weight_vector)
            if w_sum > 0:
                weight_vector /= w_sum

        elif self.imputation == "zero_risk":
            risk_vector[self.is_only_orig] = 0.0
        
        # 3. ADR Score = Sum(Risk * Weight)
        final_scores = risk_vector * weight_vector
        total_adr = np.sum(final_scores)
        end = time.time()
        print(f"ADR calculated successfully! (Elapsed time: {end - start:.2f}s)")
        print("-"*80)
        print(f"Total elapsed time: {self.cond_time +(end - start):.2f}s")
        print("="*80)

        if self.show_all:
            score_df = pd.DataFrame({"key": self.all_keys, "score": final_scores}).sort_values("key").reset_index(drop=True)
            return total_adr, score_df

        return total_adr


    def evaluate(self): 
        
        """
        Evaluates Attribute Disclosure Risk (ADR) using two standardized metrics.

        Returns:
         - diff_adr (float): Differential ADR (DADR). The absolute reduction in disclosure risk achieved by replacing original data with synthetic data.
         - adr_ratio (float): ADR Ratio (ADRR). The proportion of risk retained in the synthetic data relative to the original data.
        """
        
        self.show_all = False

        base_adr = self.calculate(self.data, self.data)
        adr = self.calculate()
        
        diff_adr = base_adr - adr
        adr_ratio = adr / base_adr

        return diff_adr, adr_ratio
