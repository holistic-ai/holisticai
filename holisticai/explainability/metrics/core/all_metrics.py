from __future__ import annotations

import numpy as np
import pandas as pd

from holisticai.explainability.metrics.global_importance._contrast_metrics import (
    importance_order_constrast,
)


def position_parity(
    feature_importance: pd.DataFrame,
    conditional_feature_importance: list[pd.DataFrame],
):
    """
    Parameters
    ----------
    feature_importance: pandas dataframe
        dataframe with feature importances
    conditional_feature_importance: list of dataframes
        list of dataframes with conditional feature importances

    Returns
    -------
    float
        position parity value
    """
    return np.mean(
        [
            importance_order_constrast(
                feature_importance_indexes=feature_importance.index,
                conditional_features_importance_indexes=i.index,
            )
            for i in conditional_feature_importance
        ]
    )
