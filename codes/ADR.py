import risk_weight_fns as rwf
import pandas as pd
import numpy as np
from conditional_dist import compute_conditional_distributions

#=======================================================================================================================
# A UNIFIED FRAMEWORK FOR ATTRIBUTE DISCLOSURE RISK (ADR) 
#
# Inputs:
# - data: Original dataset (DataFrame)
# - syn_data: Synthetic data generated from the original data (DataFrame)
# - key: Conditioning columns (list[str] or str)
# - target: Target columns containing sensitive information (list[str] or str)
# - risk: Risk function for measuring ADR (str)
# - weight: Weight function for measuring ADR (str)
# - risk_imputation: Type of imputation for unmatched keys (None or str)
# - normalize: Whether to normalize concentration-based weight functions (bool)
# - show_all: Whether to return individual key-level values or only the final ADR (bool)
#
# Outputs:
# - ADR_score: The final aggregated ADR value (float)
# - score_df: Detailed results for each key, returned when show_all is True (DataFrame or None)
#=======================================================================================================================


class ADR:
    def __init__(self, 
                 data,
                 syn_data, 
                 key, 
                 target, 
                 risk = "inner_product_similarity", 
                 weight = "weight_key_proportion1", 
                 imputation = None, 
                 normalize = True, 
                 show_all = False, 
                 **kwargs):
        
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
        
        self.ADR_score = 0.0
        self.score_df = []

        allowed_risk_nms = ["inner_product_similarity", 
                            "cosine_similarity", 
                            "bhattacharyya_coefficient", 
                            "total_variation", 
                            "hellinger_distance", 
                            "KL_divergence", 
                            "JS_divergence", 
                            "wasserstein_distance", 
                            "accuracy", 
                            "tcap_similarity", 
                            "mode_similarity", 
                            "precision", 
                            "recall"]

        allowed_weight_nms = ["weight_key_proportion1", 
                              "weight_key_proportion2", 
                              "weight_tcap", 
                              "weight_precision", 
                              "negentropy", 
                              "gini_impurity"]

        allowed_imputation_nms = [None, 
                                  "zero_risk", 
                                  "discard", 
                                  "naive", 
                                  "appr"]

        if self.risk not in allowed_risk_nms:
            raise ValueError(
                f"Invalid risk: '{self.risk}'. "
                f"You must choose one of {allowed_risk_nms}."
            )

        if self.weight not in allowed_weight_nms:
            raise ValueError(
                f"Invalid weight: '{self.weight}'. "
                f"You must choose one of {allowed_weight_nms}."
            )


        if self.imputation not in allowed_imputation_nms:
            raise ValueError(
                f"Invalid risk_imputation: '{self.imputation}'. "
                f"You must choose one of {allowed_imputation_nms}."
            )

    
    def _precompute_global_stats(self):
        prob_col = 'imputed_prob' if self.imputation in ["naive", "appr"] else 'cond_prob'
        
        self.deterministic_keys2 = self.syn_cond_dist.loc[self.syn_cond_dist[prob_col] == 1.0, 'composite_key'].tolist()
        self.best_targets_df = self.syn_cond_dist.loc[self.syn_cond_dist.groupby("composite_key")[prob_col].idxmax(), ['composite_key', 'composite_target']]
        
    
    def prepare_key_params(self, k):        
        key_df1 = self.orig_cond_dist[self.orig_cond_dist.composite_key == k]
        p1 = key_df1["cond_prob"]
        mode1 = key_df1.loc[key_df1["cond_prob"].idxmax(), "composite_target"]
      
        key_df2 = self.syn_cond_dist[self.syn_cond_dist.composite_key == k]
        prob_col = 'imputed_prob' if self.imputation in ["naive", "appr"] else 'cond_prob'
        p2 = key_df2[prob_col]
        mode2 = key_df2.loc[key_df2[prob_col].idxmax(), "composite_target"]

        params = {
            "key": k,
            "cond_dist1": self.orig_cond_dist, 
            "cond_dist2": self.syn_cond_dist,
            "key_df1": key_df1,
            "key_df2": key_df2,
            "p1": p1,
            "p2": p2,
            "mode1": mode1,
            "mode2": mode2, 
            "deterministic_keys2": self.deterministic_keys2, 
            "best_targets_df": self.best_targets_df,
            "normalize": self.normalize,
            **self.configs
        }

        return params

   
    def calculate(self):
        self.orig_cond_dist, self.syn_cond_dist = compute_conditional_distributions(self.data, 
                                                                                    self.syn_data, 
                                                                                    self.key, 
                                                                                    self.target, 
                                                                                    self.imputation, 
                                                                                    **self.configs)
        self._precompute_global_stats()
        
        orig_key_counts = self.orig_cond_dist.groupby("composite_key")["count"].sum() 
        syn_key_counts = self.syn_cond_dist.groupby("composite_key")["count"].sum()

        intersection_keys = orig_key_counts[(orig_key_counts != 0) & (syn_key_counts != 0)].index.tolist()
        only_orig_keys = orig_key_counts[syn_key_counts == 0].index.tolist()
        only_syn_keys = syn_key_counts[orig_key_counts == 0].index.tolist()
        
        risk_func = getattr(rwf, self.risk)
        weight_func = getattr(rwf, self.weight)

        intersection_weight_sum = 0.0
        
        # -------------------------------------------------------------------------------------------------
        # Intersection Keys
        # -------------------------------------------------------------------------------------------------
        for k in intersection_keys:
            params = self.prepare_key_params(k)
            
            r_val = risk_func(**params)
            w_val = weight_func(**params)
            
            score = r_val * w_val
            self.ADR_score += score
            intersection_weight_sum += w_val

            if self.show_all:
                self.score_df.append({"key": k, "score": score})

        # -------------------------------------------------------------------------------------------------
        # Only Original Keys
        # -------------------------------------------------------------------------------------------------
        if self.imputation == "discard":
            if intersection_weight_sum > 0:
                self.ADR_score = self.ADR_score / intersection_weight_sum
                
                if self.show_all:
                    for res in self.score_df:
                        res["score"] = res["score"] / intersection_weight_sum
                        
                    for k in only_orig_keys:
                        self.score_df.append({"key": k, "score": 0.0})
            
            else:
                self.ADR_score = 0.0
                if self.show_all:
                    for res in self.score_df:
                        res["score"] = 0.0

        
        elif self.imputation == "zero_risk":
            if show_all:
                for k in only_orig_keys:
                    self.score_df.append({"key": k, "score": 0.0})
                    

        else:
            for k in only_orig_keys:
                params = self.prepare_key_params(k)
                
                r_val = risk_func(**params)
                w_val = weight_func(**params)
                
                score = r_val * w_val
                self.ADR_score += score
                
                if self.show_all:
                    self.score_df.append({"key": k, "score": score})
            
      

        # for key in only_syn_keys:
        #     self.score = 0.0
        #     self.ADR_score += self.score

        #     if self.show_all:
        #         self.key_score.append({"key": key, "score": self.score})

        if self.show_all:
            self.score_df = pd.DataFrame(self.score_df)
            self.score_df = self.score_df.sort_values(by = "key", ascending = True).reset_index(drop = True)
            return self.ADR_score, self.score_df

        return self.ADR_score
