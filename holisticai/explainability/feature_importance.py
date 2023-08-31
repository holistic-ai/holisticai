import pandas as pd

from holisticai.explainability.metrics.global_importance import (
    fourth_fifths,
    global_explainability_score,
    global_overlap_score,
    global_range_overlap_score,
    importance_spread_divergence,
    importance_spread_ratio,
    surrogate_efficacy,
)
from holisticai.explainability.metrics.local_importance import (
    dataset_spread_stability,
    features_spread_stability,
)
from holisticai.explainability.metrics.utils import get_top_k, get_top_k_lime


class GlobalFeatureImportance:
    pass


class LocalFeatureImportance:
    pass


class BaseFeatureImportance:
    def __init__(
        self,
        model_type,
        model,
        x,
        y,
        importance_weights,
        conditional_importance_weights,
    ):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.importance_weights = importance_weights
        self.conditional_importance_weights = conditional_importance_weights

    def custom_metrics(self):
        pass


class PermutationFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(
        self,
        model_type,
        model,
        x,
        y,
        importance_weights,
        conditional_importance_weights,
    ):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights

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

    def metrics(self, feature_importance, conditional_feature_importance):

        reference_values = {
            "Fourth Fifths": 0,
            "Importance Spread Divergence": "-",
            "Importance Spread Ratio": 0,
            "Global Overlap Score [label=0]": 1,
            "Global Range Overlap Score [label=0]": 1,
            "Global Overlap Score [label=1]": 1,
            "Global Range Overlap Score [label=1]": 1,
            "Global Overlap Score [Q0-Q1]": 1,
            "Global Overlap Score [Q1-Q2]": 1,
            "Global Overlap Score [Q2-Q3]": 1,
            "Global Overlap Score [Q3-Q4]": 1,
            "Global Range Overlap Score [Q0-Q1]": 1,
            "Global Range Overlap Score [Q1-Q2]": 1,
            "Global Range Overlap Score [Q2-Q3]": 1,
            "Global Range Overlap Score [Q3-Q4]": 1,
            "Global Explainability Score": 1,
        }

        metrics = pd.concat(
            [
                fourth_fifths(feature_importance),
                importance_spread_divergence(feature_importance),
                importance_spread_ratio(feature_importance),
                global_overlap_score(
                    feature_importance, conditional_feature_importance
                ),
                global_range_overlap_score(
                    feature_importance, conditional_feature_importance
                ),
                global_explainability_score(
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


class SurrogateFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(self, model_type, model, x, y, importance_weights, surrogate):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.feature_importance = importance_weights
        self.surrogate = surrogate

    def get_topk(self, top_k):
        if top_k is None:
            feat_imp = self.feature_importance
        else:
            feat_imp = get_top_k(self.feature_importance, top_k)

        return {"feature_importance": feat_imp}

    def metrics(self, feature_importance):

        reference_values = {
            "Fourth Fifths": 0,
            "Importance Spread Divergence": "-",
            "Importance Spread Ratio": 0,
            "Global Explainability Score": 1,
            "Surrogate Efficacy Classification": 1,
            "Surrogate Efficacy Regression": 0,
        }

        metrics = pd.concat(
            [
                fourth_fifths(feature_importance),
                importance_spread_divergence(feature_importance),
                importance_spread_ratio(feature_importance),
                global_explainability_score(
                    self.model_type, self.model, self.x, self.y, feature_importance
                ),
                surrogate_efficacy(self.model_type, self.x, self.y, self.surrogate),
            ],
            axis=0,
        )

        reference_column = pd.DataFrame(
            [reference_values.get(metric) for metric in metrics.index],
            columns=["Reference"],
        ).set_index(metrics.index)
        metrics_with_reference = pd.concat([metrics, reference_column], axis=1)

        return metrics_with_reference


class LimeFeatureImportance(BaseFeatureImportance, LocalFeatureImportance):
    def __init__(self, importance_weights, conditional_importance_weights):
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights

    def get_topk(self, top_k):
        if top_k is None:
            feat_imp = self.feature_importance
            cond_feat_imp = self.conditional_feature_importance
        else:
            feat_imp = get_top_k_lime(self.feature_importance, top_k)
            cond_feat_imp = {
                label: get_top_k_lime(value, top_k)
                for label, value in self.conditional_feature_importance.items()
            }

        return {
            "feature_importance": feat_imp,
            "conditional_feature_importance": cond_feat_imp,
        }

    def metrics(self, feature_importance, conditional_feature_importance):

        reference_values = {
            "Features Spread Stability": 0,
            "Features Spread Ratio": 0,
            "Features Spread Mean": 0,
            "Dataset Spread Stability": 0,
            "Dataset Spread Ratio": 0,
            "Dataset Spread Mean": 0,
        }

        metrics = pd.concat(
            [
                dataset_spread_stability(
                    feature_importance, conditional_feature_importance
                )["result"],
                features_spread_stability(
                    feature_importance, conditional_feature_importance
                )["result"],
            ],
            axis=0,
        )

        reference_column = pd.DataFrame(
            [reference_values.get(metric) for metric in metrics.index],
            columns=["Reference"],
        ).set_index(metrics.index)
        metrics_with_reference = pd.concat([metrics, reference_column], axis=1)

        return metrics_with_reference
