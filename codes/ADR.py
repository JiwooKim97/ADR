from risk_weight_fns import risk_weight_fns
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
    def __init__(self, data, syn_data, key, target, risk = "inner_product_similarity", weight = "weight_key_proportion1", risk_imputation = None, normalize = True, show_all = False, **kwargs):
        self.data = data
        self.syn_data = syn_data
        self.key = key
        self.target = target

        allowed_risk_nms = ["inner_product_similarity", "cosine_similarity", "bhattacharyya_coefficient", "total_variation", "hellinger_distance", "KL_divergence", "JS_divergence", "wasserstein_distance",
                           "accuracy", "tcap_similarity", "mode_similarity",
                           "precision", "recall"]

        allowed_weight_nms = ["weight_key_proportion1", "weight_key_proportion2", "weight_tcap", "weight_precision",
                             "negentropy", "gini_impurity"]

        allowed_risk_imputation_nms = [None, "zero_risk", "discard", "naive", "appr"]

        if risk not in allowed_risk_nms:
            raise ValueError(
                f"Invalid risk: '{risk}'. "
                f"You must choose one of {allowed_risk_nms}."
            )

        if weight not in allowed_weight_nms:
            raise ValueError(
                f"Invalid weight: '{weight}'. "
                f"You must choose one of {allowed_weight_nms}."
            )


        if risk_imputation not in allowed_risk_imputation_nms:
            raise ValueError(
                f"Invalid risk_imputation: '{risk_imputation}'. "
                f"You must choose one of {allowed_risk_imputation_nms}."
            )

        self.risk = risk
        self.weight = weight
        self.risk_imputation = risk_imputation
        self.normalize = normalize
        self.show_all = show_all

        self.neighborhood = None
        if self.risk_imputation == "appr":
            self.neighborhood = kwargs.get('neighborhood')
            
            if self.neighborhood is None:
                self.neighborhood = 1
                
                import warnings
                warnings.warn(
                    f"For imputation='{self.risk_imputation}', a 'neighborhood' value was not provided. "
                    f"Using default value of 1." )

        self.orig_cond_dist, self.syn_cond_dist = compute_conditional_distributions(self.data, self.syn_data, self.key, self.target, self.risk_imputation, self.neighborhood)
               
        self.ADR_score = 0.0
        self.score_df = []

        orig_key_counts = self.orig_cond_dist.groupby("composite_key")["count"].sum() 
        syn_key_counts = self.syn_cond_dist.groupby("composite_key")["count"].sum()

        intersection_keys = orig_key_counts[(orig_key_counts != 0) & (syn_key_counts != 0)].index.tolist()
        only_orig_keys = orig_key_counts[syn_key_counts == 0].index.tolist()
        only_syn_keys = syn_key_counts[orig_key_counts == 0].index.tolist()

        
        ## intersection keys between orig_composite_keys and syn_composite_keys
        for k in intersection_keys:
            risk_weight_fn = risk_weight_fns(self.orig_cond_dist, self.syn_cond_dist, composite_key_value = k, imputation = None, normalize = self.normalize)
            result = risk_weight_fn.calculate(risk_type = self.risk, weight_type = self.weight)
            self.ADR_score += result["score"]

            if self.show_all:
                self.score_df.append(result)
        
        for k in only_orig_keys:
            risk_weight_fn = risk_weight_fns(self.orig_cond_dist, self.syn_cond_dist, composite_key_value = k, imputation = self.risk_imputation, normalize = self.normalize)
            result = risk_weight_fn.calculate(risk_type = self.risk, weight_type = self.weight)

            if self.risk_imputation == "discard":
                numerator = self.ADR_score
                denominator = np.sum(self.orig_cond_dist[(self.orig_cond_dist["composite_key"].isin(intersection_keys))]["count"])/ np.sum(self.orig_cond_dist["count"])
                risk = numerator / denominator
                
                weight = getattr(risk_weight_fn, self.weight)()
                score = risk * weight
                self.ADR_score += score

                if self.show_all:
                    self.score_df.append({"key": k, "score": score})
            else:
                self.ADR_score += result["score"]
                
                if self.show_all:
                    self.score_df.append(result)
      

        # for key in only_syn_keys:
        #     self.score = 0.0
        #     self.ADR_score += self.score

        #     if self.show_all:
        #         self.key_score.append({"key": key, "score": self.score})

        if self.show_all:
            self.score_df = pd.DataFrame(self.score_df)
            self.score_df = self.score_df.sort_values(by = "key", ascending = True).reset_index(drop = True) 
