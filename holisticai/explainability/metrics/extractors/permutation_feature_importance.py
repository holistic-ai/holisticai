import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from ..global_importance import (
    ExplainabilityEase,
    FourthFifths,
    ImportantSimilarity,
    PositionParity,
    RankAlignment,
    SpreadDivergence,
    SpreadRatio,
)
from ..utils import BaseFeatureImportance, GlobalFeatureImportance, get_index_groups


class NormalizedFeatureImportance:
    def __init__(self, config):
        self.default_max_samples = config.pop("max_samples")
        self.config = config

    def __call__(self, model, x, y):
        max_samples = min(x.shape[0], self.default_max_samples)
        feat_imp = permutation_importance(
            model, x, y, n_jobs=-1, max_samples=max_samples, **self.config
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


def compute_permutation_feature_importance(model_type, x, y, **kargs):

    model = kargs.get("model", None)

    fi_config = {
        "max_samples": kargs.get("max_samples", 3000),
        "n_repeats": kargs.get("n_repeats", 5),
        "random_state": kargs.get("random_state", 42),
    }
    norm_importance = NormalizedFeatureImportance(fi_config)

    # Feature Importance
    features_importance = norm_importance(model, x, y)

    # Conditional Feature Importance (classification:category, regression:quantile)
    index_groups = get_index_groups(model_type, y)
    conditional_features_importance = {
        str(label): norm_importance(model, x.loc[index], y.loc[index])
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

    def metrics(self, alpha, detailed, metric_names):
        if metric_names is None:
            metric_names = [
                "Explainability Ease",
                "Fourth Fifths",
                "Position Parity",
                "Rank Alignment",
                "Region Similarity",
                "Spread Divergence",
                "Spread Ratio",
            ]

        (feat_imp, cond_feat_imp), (
            alpha_feat_imp,
            alpha_cond_feat_imp,
        ) = self.get_alpha_feature_importance(alpha)

        spread_metrics = {
            "Spread Divergence": SpreadDivergence(detailed=detailed),
            "Spread Ratio": SpreadRatio(detailed=detailed),
        }

        position_metrics = {
            "Position Parity": PositionParity(detailed=detailed),
            "Rank Alignment": RankAlignment(detailed=detailed),
        }

        metric_scores = []

        if "Fourth Fifths" in metric_names:
            ff = FourthFifths(detailed=detailed)
            scores = ff(feat_imp, cond_feat_imp)
            metric_scores += [
                {"Metric": metric_name, "Value": value, "Reference": ff.reference}
                for metric_name, value in scores.items()
            ]

        for metric_name, metric_fn in spread_metrics.items():
            if metric_name in metric_names:
                scores = metric_fn(feat_imp, cond_feat_imp)
                metric_scores += [
                    {
                        "Metric": metric_score_name,
                        "Value": value,
                        "Reference": metric_fn.reference,
                    }
                    for metric_score_name, value in scores.items()
                ]

        for metric_name, metric_fn in position_metrics.items():
            if metric_name in metric_names:
                scores = metric_fn(alpha_feat_imp, cond_feat_imp)
                metric_scores += [
                    {
                        "Metric": metric_score_name,
                        "Value": value,
                        "Reference": metric_fn.reference,
                    }
                    for metric_score_name, value in scores.items()
                ]

        if "Important Similarity" in metric_names:
            imp_sim = ImportantSimilarity(detailed=detailed)
            scores = imp_sim(feat_imp, cond_feat_imp)
            metric_scores += [
                {"Metric": metric_name, "Value": value, "Reference": expe.reference}
                for metric_name, value in scores.items()
            ]

        if "Explainability Ease" in metric_names:
            expe = ExplainabilityEase(
                model_type=self.model_type, model=self.model, x=self.x
            )
            scores = expe(alpha_feat_imp)
            metric_scores += [
                {"Metric": metric_name, "Value": value, "Reference": expe.reference}
                for metric_name, value in scores.items()
            ]

        return pd.DataFrame(metric_scores).set_index("Metric").sort_index()
