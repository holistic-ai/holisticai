from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from holisticai.xai.commons._definitions import ConditionalFeatureImportance, FeatureImportance


class RankAlignment(BaseModel):
    name: str = "Rank Alignment"
    reference: float = 1.0

    def __call__(
        self, conditional_feature_importance: ConditionalFeatureImportance, feature_importance: FeatureImportance
    ):
        feature_names = feature_importance.feature_names
        conditional_position_parity = {}
        for group_name, cond_features in conditional_feature_importance.conditional_feature_importance.items():
            cond_feature_names = cond_features.feature_names
            intersections = []
            for top_k in range(1, len(feature_importance) + 1):
                ggg = set(cond_feature_names[:top_k])
                vvv = set(feature_names[:top_k])
                u = len(set(ggg).intersection(vvv)) / top_k
                intersections.append(u)
            conditional_position_parity[group_name] = np.mean(intersections)
        return np.mean(np.mean(list(conditional_position_parity.values())))


def rank_alignment(conditional_feature_importance, ranked_feature_importance):
    metric = RankAlignment()
    return metric(conditional_feature_importance, ranked_feature_importance)
