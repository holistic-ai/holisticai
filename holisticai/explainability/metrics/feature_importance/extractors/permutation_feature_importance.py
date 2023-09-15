import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)

from ..global_importance import (
    fourth_fifths,
    global_explainability_ease_score,
    global_overlap_score,
    global_range_overlap_score,
    global_similarity_score,
    importance_spread_divergence,
    importance_spread_ratio,
)
from .extractor_utils import (
    BaseFeatureImportance,
    GlobalFeatureImportance,
    get_index_groups,
    get_top_k,
)


def feature_importance(model, x, y):
    n_repeats = 5
    random_state = 42
    max_samples = min(x.shape[0], 1000)
    feat_imp = permutation_importance(
        model,
        x,
        y,
        n_jobs=-1,
        n_repeats=n_repeats,
        random_state=random_state,
        max_samples=max_samples,
    )
    df_feat_imp = pd.DataFrame(
        {
            "Variable": x.columns,
            "Importance": feat_imp["importances_mean"],
            "Std": feat_imp["importances_std"],
        }
    )
    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp["Importance"] /= df_feat_imp["Importance"].sum()
    df_feat_imp = df_feat_imp.sort_values("Importance", ascending=False).copy()

    return df_feat_imp


def compute_permutation_feature_importance(model_type, model, x, y):
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    if not isinstance(y, pd.Series):
        y = _array_like_to_series(y)

    # Feature Importance
    features_importance = feature_importance(model, x, y)

    # Conditional Feature Importance (classification:category, regression:quantile)
    index_groups = get_index_groups(model_type, y)
    conditional_features_importance = {
        str(label): feature_importance(model, x.iloc[index], y.iloc[index])
        for label, index in index_groups.items()
    }

    return PermutationFeatureImportance(
        model_type,
        model,
        x,
        y,
        features_importance,
        conditional_features_importance,
        index_groups,
    )


class PermutationFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(
        self,
        model_type,
        model,
        x,
        y,
        importance_weights,
        conditional_importance_weights,
        index_groups,
    ):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights
        self.index_groups = index_groups

    def get_topk(self, top_k):
        if top_k is None:
            feat_imp = self.feature_importance
            cond_feat_imp = self.conditional_feature_importance
        else:
            feat_imp = get_top_k(self.feature_importance, top_k)
            cond_feat_imp = {
                label: get_top_k(value, top_k)
                for label, value in self.conditional_feature_importance.items()
            }

        return {
            "feature_importance": feat_imp,
            "conditional_feature_importance": cond_feat_imp,
        }

    def metrics(self, feature_importance, conditional_feature_importance, detailed):

        reference_values = {
            "Fourth Fifths": 0,
            "Importance Spread Divergence": "-",
            "Importance Spread Ratio": 0,
            "Global Overlap Score [label=0]": 1,
            "Global Range Overlap Score [label=0]": 1,
            "Global Overlap Score [label=1]": 1,
            "Global Range Overlap Score [label=1]": 1,
            "Global Overlap Score": 1,
            "Global Overlap Score [Q0-Q1]": 1,
            "Global Overlap Score [Q1-Q2]": 1,
            "Global Overlap Score [Q2-Q3]": 1,
            "Global Overlap Score [Q3-Q4]": 1,
            "Global Range Overlap Score": 1,
            "Global Range Overlap Score [Q0-Q1]": 1,
            "Global Range Overlap Score [Q1-Q2]": 1,
            "Global Range Overlap Score [Q2-Q3]": 1,
            "Global Range Overlap Score [Q3-Q4]": 1,
            "Global Similarity Score": 1,
            "Global Similarity Score [Q0-Q1]": 1,
            "Global Similarity Score [Q1-Q2]": 1,
            "Global Similarity Score [Q2-Q3]": 1,
            "Global Similarity Score [Q3-Q4]": 1,
            "Global Explainability Ease Score": 1,
        }

        metrics = pd.concat(
            [
                fourth_fifths(feature_importance),
                importance_spread_divergence(feature_importance),
                importance_spread_ratio(feature_importance),
                global_overlap_score(
                    feature_importance, conditional_feature_importance, detailed
                ),
                global_range_overlap_score(
                    feature_importance, conditional_feature_importance, detailed
                ),
                global_similarity_score(
                    feature_importance, conditional_feature_importance, detailed
                ),
                global_explainability_ease_score(
                    self.model_type, self.model, self.x, self.y, feature_importance
                ),
            ],
            axis=0,
        )

        reference_column = pd.DataFrame(
            [reference_values.get(metric) for metric in metrics.index],
            columns=["Reference"],
        ).set_index(metrics.index)
        metrics_with_reference = pd.concat([metrics, reference_column], axis=1)

        return metrics_with_reference
