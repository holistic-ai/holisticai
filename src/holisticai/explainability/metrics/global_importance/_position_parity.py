from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from holisticai.explainability.commons._definitions import ConditionalFeatureImportance, Importances


class PositionParity(BaseModel):
    name: str = "Position Parity"
    reference: float = 1.0

    def __call__(self, conditional_feature_importance: ConditionalFeatureImportance, feature_importance: Importances):
        conditional_position_parity = {}
        for group_name, cond_features in conditional_feature_importance:
            match_order = [c == r for c, r in zip(cond_features.feature_names, feature_importance.feature_names)]
            m_order_cum = np.cumsum(match_order) / np.arange(1, len(match_order) + 1)
            conditional_position_parity[group_name] = np.mean(m_order_cum)
        return np.mean(np.mean(list(conditional_position_parity.values())))


def position_parity(
    conditional_feature_importance: ConditionalFeatureImportance, ranked_feature_importance: Importances
):
    """
    Calculates the position parity metric.

    This metric measures the difference between the conditional feature importance and the ranked feature importance.
    It quantifies how much the ranking of feature importance changes when considering conditional importance.

    Parameters
    ----------
    conditional_feature_importance: ConditionalFeatureImportance
        A list of conditional feature importance values.
    ranked_feature_importance: Importances
        A list of ranked feature importance values.

    Returns
    -------
    float
        The position parity metric value.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.explainability.commons import (
    ...     ConditionalFeatureImportance,
    ...     Importances,
    ... )
    >>> from holisticai.explainability.metrics import position_parity
    >>> values = np.array([0.50, 0.40, 0.10])
    >>> feature_names = ["feature_1", "feature_2", "feature_3"]
    >>> feature_importance = Importances(values=values, feature_names=feature_names)
    >>> values = {
    ...     "group1": Importances(
    ...         values=np.array([0.40, 0.35, 0.25]),
    ...         feature_names=["feature_1", "feature_2", "feature_3"],
    ...     ),
    ...     "group2": Importances(
    ...         values=np.array([0.50, 0.30, 0.20]),
    ...         feature_names=["feature_3", "feature_2", "feature_1"],
    ...     ),
    ... }
    >>> conditional_feature_importance = ConditionalFeatureImportance(values=values)
    >>> position_parity(conditional_feature_importance, feature_importance)
    0.6388888888888888
    """
    metric = PositionParity()
    return metric(conditional_feature_importance, ranked_feature_importance)
