from __future__ import annotations

import numpy as np
import pandas as pd

from holisticai.explainability.metrics.global_importance._contrast_metrics import (
    importance_order_constrast,
    importance_range_constrast,
)
from holisticai.explainability.metrics.global_importance._explainability_level import (
    compute_explainability_ease_score,
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


def rank_alignment(
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
        rank alignment value
    """
    return np.mean(
        [
            importance_range_constrast(
                feature_importance_indexes=feature_importance.index,
                conditional_features_importance_indexes=i.index,
            )
            for i in conditional_feature_importance
        ]
    )


def explainability_ease(partial_dependence_list: list[dict]):
    """
    Parameters
    ----------
    partial_dependence_list: list[dict]
        a list of dictionaries containing partial dependencies for each feature.
        For multiclass classification, partial dependencies are computed for each class separately.
        For binary classification, partial dependence is calculated only for the positive class,
        resulting in a single dictionary in the list. Similarly, for regression, there's only one dictionary.

    Returns
    -------
    float
        explainability ease value, average explainability ease value for multiclass setting
    """
    if len(partial_dependence_list) == 1:
        return compute_explainability_ease_score(
            partial_dependence=partial_dependence_list[0]
        )[0]
    else:
        return np.mean(
            [
                compute_explainability_ease_score(
                    partial_dependence=partial_dependence
                )[0]
                for partial_dependence in partial_dependence_list
            ]
        )

