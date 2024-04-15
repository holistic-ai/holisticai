from __future__ import annotations

from typing import Iterable

import numpy as np

from holisticai.explainability.metrics.global_importance._contrast_metrics import importance_order_constrast,importance_range_constrast

def position_parity(
    feature_importance_indexes: Iterable,
    conditional_features_importance_indexes: list[Iterable],
):
    return np.mean(
        [
            importance_order_constrast(
                feature_importance_indexes=feature_importance_indexes,
                conditional_features_importance_indexes=i,
            )
            for i in conditional_features_importance_indexes
        ]
    )


def rank_alignment(
    feature_importance_indexes: Iterable,
    conditional_features_importance_indexes: list[Iterable],
):
    return np.mean(
        [
            importance_range_constrast(
                feature_importance_indexes=feature_importance_indexes,
                conditional_features_importance_indexes=i,
            )
            for i in conditional_features_importance_indexes
        ]
    )
