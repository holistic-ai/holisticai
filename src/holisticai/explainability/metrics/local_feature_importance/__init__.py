import pandas as pd

from holisticai.explainability.metrics.local_feature_importance._importance_stability import importance_stability
from holisticai.explainability.metrics.local_feature_importance._rank_consistency import (
    local_normalized_desviation,
    rank_consistency,
)
from holisticai.explainability.metrics.local_feature_importance._stability import (
    FeatureStability,
    compute_importance_distribution,
    feature_stability,
)

__all__ = [
    "FeatureStability",
    "feature_stability",
    "compute_importance_distribution",
    "importance_stability",
    "rank_consistency",
    "local_normalized_desviation",
]


def classification_local_feature_importance_explainability_metrics(local_importances):
    metrics = pd.DataFrame(index=["Rank Consistency", "Importance Stability"], columns=["Value", "Reference"])
    metrics.at["Rank Consistency", "Value"] = rank_consistency(local_importances.values)
    metrics.at["Rank Consistency", "Reference"] = 0

    metrics.at["Importance Stability", "Value"] = importance_stability(local_importances.values)
    metrics.at["Importance Stability", "Reference"] = 0
    return metrics


def regression_local_feature_importance_explainability_metrics(local_importances):
    metrics = pd.DataFrame(index=["Rank Consistency", "Importance Stability"], columns=["Value", "Reference"])
    metrics.at["Rank Consistency", "Value"] = rank_consistency(local_importances.values)
    metrics.at["Rank Consistency", "Reference"] = 0

    metrics.at["Importance Stability", "Value"] = importance_stability(local_importances.values)
    metrics.at["Importance Stability", "Reference"] = 0
    return metrics
