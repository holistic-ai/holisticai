import pandas as pd

from holisticai.explainability.metrics.global_feature_importance._alpha_score import (
    AlphaScore,
    alpha_score,
)
from holisticai.explainability.metrics.global_feature_importance._fluctuation_ratio import fluctuation_ratio
from holisticai.explainability.metrics.global_feature_importance._importance_spread import (
    SpreadDivergence,
    SpreadRatio,
    spread_divergence,
    spread_ratio,
)
from holisticai.explainability.metrics.global_feature_importance._position_parity import (
    PositionParity,
    position_parity,
)
from holisticai.explainability.metrics.global_feature_importance._rank_alignment import RankAlignment, rank_alignment
from holisticai.explainability.metrics.global_feature_importance._surrogate import (
    surrogate_accuracy_score,
    surrogate_mean_squared_error,
)
from holisticai.explainability.metrics.global_feature_importance._xai_ease_score import XAIEaseScore, xai_ease_score


def classification_global_feature_importance_explainability_metrics(
    partial_dependencies, importances, conditional_feature_importances, top_n
):
    metrics = pd.DataFrame(
        index=["Spread Divergence", "Fluctuation Ratio", "Rank Alignment"], columns=["Value", "Reference"]
    )

    metrics.at["Alpha Score", "Value"] = alpha_score(importances)
    metrics.at["Alpha Score", "Reference"] = 0

    metrics.at["Spread Divergence", "Value"] = spread_divergence(importances)
    metrics.at["Spread Divergence", "Reference"] = 1

    metrics.at["Fluctuation Ratio", "Value"] = fluctuation_ratio(partial_dependencies, importances, top_n=top_n)
    metrics.at["Fluctuation Ratio", "Reference"] = 0

    metrics.at["Rank Alignment", "Value"] = rank_alignment(conditional_feature_importances, importances)
    metrics.at["Rank Alignment", "Reference"] = 1

    return metrics


__all__ = [
    "fluctuation_ratio",
    "AlphaScore",
    "XAIEaseScore",
    "PositionParity",
    "RankAlignment",
    "SpreadRatio",
    "SpreadDivergence",
    "alpha_score",
    "xai_ease_score",
    "position_parity",
    "rank_alignment",
    "spread_ratio",
    "spread_divergence",
    "surrogate_accuracy_score",
    "surrogate_mean_squared_error",
]
