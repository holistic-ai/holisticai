from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from holisticai.xai.commons._definitions import ConditionalFeatureImportance, FeatureImportance


class PositionParity(BaseModel):
    name: str = "Position Parity"
    reference: float = 1.0

    def __call__(
        self, conditional_feature_importance: ConditionalFeatureImportance, feature_importance: FeatureImportance
    ):
        conditional_position_parity = {}
        for group_name, cond_features in conditional_feature_importance.conditional_feature_importance.items():
            match_order = [c == r for c, r in zip(cond_features.feature_names, feature_importance.feature_names)]
            m_order_cum = np.cumsum(match_order) / np.arange(1, len(match_order) + 1)
            conditional_position_parity[group_name] = np.mean(m_order_cum)
        return np.mean(np.mean(list(conditional_position_parity.values())))


def position_parity(conditional_feature_importance, ranked_feature_importance):
    metric = PositionParity()
    return metric(conditional_feature_importance, ranked_feature_importance)
