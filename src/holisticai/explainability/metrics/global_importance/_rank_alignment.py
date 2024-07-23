from __future__ import annotations

import numpy as np
from holisticai.utils import ConditionalImportance, Importances
from pydantic import BaseModel


class RankAlignment(BaseModel):
    name: str = "Rank Alignment"
    reference: float = 1.0

    def __call__(self, conditional_feature_importance: ConditionalImportance, feature_importance: Importances):
        feature_names = feature_importance.feature_names
        conditional_position_parity = {}
        for group_name, cond_features in conditional_feature_importance:
            cond_feature_names = cond_features.feature_names
            intersections = []
            for top_k in range(1, len(feature_importance) + 1):
                ggg = set(cond_feature_names[:top_k])
                vvv = set(feature_names[:top_k])
                u = len(set(ggg).intersection(vvv)) / top_k
                intersections.append(u)
            conditional_position_parity[group_name] = np.mean(intersections)
        return float(np.mean(np.mean(list(conditional_position_parity.values()))))


def rank_alignment(conditional_feature_importance: ConditionalImportance, ranked_feature_importance: Importances):
    """
    Calculates the rank alignment metric between conditional feature importance and ranked feature importance.

    Parameters
    ----------
    conditional_feature_importance: ConditionalFeatureImportance
        The conditional feature importance values.
    ranked_feature_importance: Importances
        The ranked feature importance values.

    Returns
    -------
        float: The rank alignment metric value.

    Example
    -------
    >>> from holisticai.explainability.commons import (
    ...     ConditionalFeatureImportance,
    ...     Importances,
    ... )
    >>> from holisticai.explainability.metrics import rank_alignment
    >>> values = {
    ...     "0": Importances(
    ...         values=[0.1, 0.2, 0.3, 0.4],
    ...         feature_names=["feature_2", "feature_3", "feature_4"],
    ...     ),
    ...     "1": Importances(
    ...         values=[0.4, 0.3, 0.2, 0.1],
    ...         feature_names=["feature_1", "feature_2", "feature_3", "feature_4"],
    ...     ),
    ... }
    >>> conditional_feature_importance = ConditionalFeatureImportance(values=values)
    >>> ranked_feature_importance = Importances(
    ...     values=[0.5, 0.3, 0.2],
    ...     feature_names=["feature_1", "feature_2", "feature_3"],
    ... )
    >>> rank_alignment(conditional_feature_importance, ranked_feature_importance)
    0.6944444444444444
    """
    metric = RankAlignment()
    return metric(conditional_feature_importance, ranked_feature_importance)
