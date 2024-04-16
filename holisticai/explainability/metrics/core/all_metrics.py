from __future__ import annotations
import numpy as np

from typing import Iterable
from numpy.typing import ArrayLike


from holisticai.explainability.metrics.global_importance._contrast_metrics import (
    importance_order_constrast,
)


def position_parity(
    feature_importance_indexes: Iterable,
    conditional_features_importance_indexes: list[Iterable],
):
    """
    Parameters
    ----------
    feature_importance_indexes: Iterable
        array with feature importance indexes
    conditional_feature_importance_indexes: list[Iterable]
        array with conditional feature importance indexes
    """
    return np.mean(
        [
            importance_order_constrast(
                feature_importance_indexes=feature_importance_indexes,
                conditional_features_importance_indexes=i,
            )
            for i in conditional_features_importance_indexes
        ]
    )
