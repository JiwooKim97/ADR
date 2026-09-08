import time

import numpy as np
import pandas as pd

import key_level_risk_weights as klrw
from conditional_dist import compute_conditional_distributions



class ADR_Measure:
    def __init__(
        self, 
        data, 
        syn_data, 
        key, 
        target, 
        risk_measure = "inner_product_similarity", 
        weight = "OD_prevalence",
        imputation="constant_risk",
        use_deterministic = False,
        normalize=True, 
        show_all=False, 
        **kwargs
    ):

        """
        Initialize the Attribute Disclosure Risk measure.

        Args:
            data (pd.DataFrame):
                Original dataset containing sensitive information.

            syn_data (pd.DataFrame):
                Synthetic dataset generated from the original data.

            key (list[str] or str):
                Column(s) used as conditioning variables (quasi-identifiers).

            target (list[str] or str):
                Target column(s) containing sensitive information.

            risk_measure (str):
                Key-level risk measure.

            weight (str):
                Key-level weighting function.

            imputation (str, optional):
                Method for handling unmatched keys.
                Options:
                    - "constant_risk"
                    - "exclusion"
                    - "marginal"
                    - "neighborhood_appr"

            use_deterministic (bool, optional):
                Whether to restrict weighting to deterministic keys.

            normalize (bool, optional):
                Whether to renormalize weights after restricting them to deterministic keys.

            show_all (bool, optional):
                Whether to return key-level risk measure, weight, and score values.

        Keyword Args:
            neighborhood (int):
                Number of neighbors used for neighborhood approximation.

            alpha (float):
                Positive scaling parameter used in KL_similarity and
                wasserstein_similarity.

            positive_target_value:
                Positive target value used for precision, recall,
                and prediction_positive.

            constant_risk_value (float):
                Risk assigned to original-only keys when
                imputation="constant_risk".
        """
                
        
        self.data = data
        self.syn_data = syn_data
        self.key = key
        self.target = target
        self.risk_measure = risk
        self.weight = weight
        self.imputation = imputation
        self.use_deterministic = use_deterministic
        self.normalize = normalize
        self.show_all = show_all
        self.configs = kwargs

        self.orig_cond_dist = None
        self.syn_cond_dist = None


    # =============================================================================================
    # Internal Precomputation
    # =============================================================================================

    def _precompute_key_target_uniques(self):
        key_cols = [self.key] if isinstance(self.key, str) else list(self.key)
        target_cols = [self.target] if isinstance(self.target, str) else list(self.target)
        
        all_keys_df = pd.concat(
            [self.data[key_cols], 
             self.syn_data[key_cols]], 
            axis=0
        )
        key_tuples = pd.MultiIndex.from_frame(all_keys_df).to_list()
        _, self.k_uniques = pd.factorize(pd.Index(key_tuples), sort=True)
        
        all_targets_df = pd.concat(
            [self.data[target_cols], 
             self.syn_data[target_cols]], 
            axis=0
        )
        target_tuples = pd.MultiIndex.from_frame(all_targets_df).to_list()
        _, self.t_uniques = pd.factorize(pd.Index(target_tuples), sort=True)


    
    def _precompute_global_stats(self):
        prob_col = (
            'imputed_prob' 
            if 'imputed_prob' in self.syn_cond_dist.columns 
            else 'cond_prob'
        )
        
        self.p1_matrix_df = (
            self.orig_cond_dist
            .pivot(
                index='composite_key', 
                columns='composite_target', 
                values='cond_prob'
            )
            .fillna(0)
        )
        
        self.p2_matrix_df = (
            self.syn_cond_dist
            .pivot(
                index='composite_key', 
                columns='composite_target', 
                values=prob_col
            )
            .fillna(0)
        )
        
        self.p1_matrix = self.p1_matrix_df.values
        self.p2_matrix = self.p2_matrix_df.values

        self.target_dist = (
            self.orig_cond_dist
            .groupby('composite_target')['count']
            .sum() 
            / self.orig_cond_dist['count'].sum()
        )
        
        self.target_vector = (
            self.target_dist
            .reindex(self.p1_matrix_df.columns)
            .fillna(0)
            .values
        )
        
        self.target_matrix = np.tile(
            self.target_vector, 
            (self.p1_matrix.shape[0], 1)
        )
        
        self.target_array = np.unique(
            self.orig_cond_dist['composite_target']
        )
        
        self.target_to_idx = {
            name: i 
            for i, name in enumerate(self.target_array)
        }
        
        self.mode1_vector = self.target_array[np.argmax(self.p1_matrix, axis=1)]
        self.mode2_vector = self.target_array[np.argmax(self.p2_matrix, axis=1)]
        
        self.deterministic_keys2 = self.syn_cond_dist.loc[self.syn_cond_dist[prob_col] == 1.0, 'composite_key'].unique().tolist()
        
        self.best_targets_df = pd.DataFrame({
            'composite_key': self.p2_matrix_df.index, 
            'composite_target': self.mode2_vector
        })



    
    # =============================================================================================
    # Data Preparation
    # =============================================================================================
        
    def prepare_data(self, data1 = None, data2 = None):
        current_data1 = data1 if data1 is not None else self.data
        current_data2 = data2 if data2 is not None else self.syn_data
        
        self._precompute_key_target_uniques()
        
        actual_imputation = self.imputation
        
        if (current_data2 is self.data) or (current_data2 is current_data1):
            actual_imputation = None
            
        cond_start = time.time()
        
        self.orig_cond_dist, self.syn_cond_dist = (
            compute_conditional_distributions(
                current_data1, 
                current_data2, 
                self.key, 
                self.target, 
                self.k_uniques,
                self.t_uniques,
                actual_imputation, 
                **self.configs
            )
        )
        
        cond_end = time.time()
        self.cond_time = cond_end - cond_start
        
        self._precompute_global_stats()


        orig_key_counts = (
            self.orig_cond_dist
            .groupby("composite_key")["count"]
            .sum()
        )
        
        syn_key_counts = (
            self.syn_cond_dist
            .groupby("composite_key")["count"]
            .sum()
        )

        self.all_keys = self.p1_matrix_df.index.values

        orig_present = (
            orig_key_counts
            .reindex(self.all_keys, fill_value = 0)
            > 0
        )

        syn_presnet = (
            syn_key_counts
            .reindex(self.all_keys, fill_value = 0)
            > 0
        )
        
        self.is_intersection = orig_present & syn_presnet
        self.is_only_orig = orig_present & ~syn_presnet
        self.is_only_syn = ~orig_present & syn_presnet

    


    # =============================================================================================
    # ADR Score Calculation
    # =============================================================================================
    
    def calculate(
        self, 
        data1 = None, 
        data2 = None, 
        risk_measure = None, 
        weight = None, 
        marginal_reference = False
    ):
        
        """
        Calculate the Attribute Disclosure Risk Score.

        Args:
            data1 (pd.DataFrame, optional):
                Reference dataset used as the baseline.

            data2 (pd.DataFrame, optional):
                Dataset evaluated against the reference dataset.

            risk_measure (str, optional):
                Key-level risk measure.

            weight (str, optional):
                Key-level weighting function.

            marginal_reference (bool, optional):
                Whether to replace the original conditional target
                distribution with the marginal target distribution.

        Returns:
            total_adr (float):
                Aggregated ADR score.

            score_df (pd.DataFrame, optional):
                Key-level risk, weight, and ADR score contribution.
                Returned only when show_all=True.
        """
        
        current_data1 = data1 if data1 is not None else self.data
        current_data2 = data2 if data2 is not None else self.syn_data

        name1 = "original data"
        name2 = "original data" if data2 is not None else "synthetic data"
        
        self.prepare_data(current_data1, current_data2)

        start = time.time()
        
        current_risk = risk_measure if risk_measure else self.risk_measure
        current_weight = weight if weight else self.weight
        
        risk_func = getattr(klrw, current_risk)
        weight_func = getattr(klrw, current_weight)

        self.disclosive_keys = (
            self.deterministic_keys2 
            if self.use_deterministic 
            else None
        )

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

        # -----------------------------------------------------------------------------------------
        # 1. Synthetic-only keys
        # -----------------------------------------------------------------------------------------
        risk_vector[self.is_only_syn] = 0.0

        
        # -----------------------------------------------------------------------------------------
        # 2. Original-only keys
        # -----------------------------------------------------------------------------------------
        if self.imputation == "exclusion":
            weight_vector[~self.is_intersection] = 0.0
            
            w_sum = np.sum(weight_vector)
            
            if w_sum > 0:
                weight_vector /= w_sum

        elif self.imputation == "constant_risk":
            const_val = self.configs.get(
                "constant_risk_value", 
                0.0
            )
            
            if const_val < 0:
                raise ValueError(
                    "constant_risk_value must be a number greater than or equal to 0."
                )
                
            risk_vector[self.is_only_orig] = const_val

        
        # -----------------------------------------------------------------------------------------
        # 3. Aggregate key-level ADR Score
        # -----------------------------------------------------------------------------------------
        final_scores = risk_vector * weight_vector
        
        total_adr_score = np.sum(final_scores)
        
        end = time.time()


        if self.show_all:
            score_df = pd.DataFrame({
                "key": self.all_keys, 
                "risk_measure": risk_vector, 
                "weight": weight_vector, 
                "score": final_scores})
            
            score_df = (
                score_df
                .sort_values("key")
                .reset_index(drop=True)
            )

            return total_adr_score, score_df

        return total_adr_score
