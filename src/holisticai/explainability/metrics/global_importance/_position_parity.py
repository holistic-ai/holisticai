from __future__ import annotations

import numpy as np
from holisticai.utils import ConditionalImportances, Importances


class PositionParity:
    name: str = "Position Parity"
    reference: float = 1.0

    def __call__(self, conditional_feature_importance: ConditionalImportances, feature_importance: Importances):
        conditional_position_parity = {}
        for group_name, cond_features in conditional_feature_importance:
            match_order = [c == r for c, r in zip(cond_features.feature_names, feature_importance.feature_names)]
            m_order_cum = np.cumsum(match_order) / np.arange(1, len(match_order) + 1)
            conditional_position_parity[group_name] = np.mean(m_order_cum)
        return float(np.mean(np.mean(list(conditional_position_parity.values()))))


def position_parity(conditional_feature_importance: ConditionalImportances, ranked_feature_importance: Importances):
    """
    This metric, ranging from 0 to 1, measures how well the top feature importances (>80%) maintain their ranking when considering conditional importance for classes (classification) or quantiles (regression).

    Parameters
    ----------
    conditional_feature_importance: ConditionalImportances
        The feature importance for each output label (classification) or output region (regression).
    ranked_feature_importance: Importances
        The ranked feature importance values.

    Returns
    -------
    float
        The position parity metric value.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.explainability.commons import (
    ...     ConditionalImportances,
    ...     Importances,
    ... )
    >>> from holisticai.explainability.metrics import position_parity
    >>> conditional_feature_importance = ConditionalImportances(
    ...     values={
    ...         "group1": Importances(
    ...             values=np.array([0.40, 0.35, 0.25]),
    ...             feature_names=["feature_1", "feature_2", "feature_3"],
    ...         ),
    ...         "group2": Importances(
    ...             values=np.array([0.50, 0.30, 0.20]),
    ...             feature_names=["feature_3", "feature_2", "feature_1"],
    ...         ),
    ...     }
    ... )
    >>> ranked_feature_importance = Importances(
    ...     values=np.array([0.50, 0.40, 0.10]),
    ...     feature_names=["feature_1", "feature_2", "feature_3"],
    ... )
    >>> position_parity(conditional_feature_importance, ranked_feature_importance)
    0.6388888888888888
    """
    metric = PositionParity()
    return metric(conditional_feature_importance, ranked_feature_importance)
