from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from holisticai.utils._definitions import ConditionalImportances, Importances


class RankAlignment:
    name: str = "Rank Alignment"
    reference: float = 1.0

    def __call__(
        self,
        conditional_feature_importance: ConditionalImportances,
        feature_importance: Importances,
        alpha=0.8,
        aggregation=True,
    ):
        top_feature_names = feature_importance.top_alpha(alpha=alpha).feature_names
        similarities = []
        for _, cond_features in conditional_feature_importance:
            top_cond_feature_names = cond_features.top_alpha(alpha=alpha).feature_names
            top_cond_feature_names = set(top_cond_feature_names)
            similarities.append(
                len(set(top_feature_names).intersection(top_cond_feature_names))
                / len(set(top_feature_names).union(top_cond_feature_names))
            )
        if aggregation:
            return float(np.mean(similarities))
        return similarities


def rank_alignment(
    conditional_feature_importance: ConditionalImportances, ranked_feature_importance: Importances, aggregation=True
):
    """

    Compute the rank alignment metric between conditional feature importance and ranked feature importance.

    Parameters
    ----------
    conditional_feature_importance : ConditionalImportances
        The conditional feature importance values.
    ranked_feature_importance : Importances
        The ranked feature importance values.
    aggregation : bool, optional
        If True, aggregate the results, by default True.

    Returns
    -------
    float
        The computed rank alignment metric.
    """
    metric = RankAlignment()
    return metric(conditional_feature_importance, ranked_feature_importance, aggregation=aggregation)
