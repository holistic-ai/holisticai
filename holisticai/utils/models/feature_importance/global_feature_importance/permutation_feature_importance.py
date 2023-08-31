import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from holisticai.explainability.feature_importance import PermutationFeatureImportance
from holisticai.explainability.metrics.utils import four_fifths_list, get_index_groups
from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)


def feature_importance(model, x, y):
    n_repeats = 5
    random_state = 42
    feat_imp = permutation_importance(
        model, x, y, n_repeats=n_repeats, random_state=random_state
    )
    df_feat_imp = pd.DataFrame(
        {
            "Variable": x.columns,
            "Importance": feat_imp["importances_mean"],
            "Std": feat_imp["importances_std"],
        }
    )
    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp = df_feat_imp.sort_values("Importance", ascending=False).copy()

    return df_feat_imp


def compute_permutation_feature_importance(model_type, model, x, y):
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    if not isinstance(y, pd.Series):
        y = _array_like_to_series(y)

    # Feature Importance
    features_importance = feature_importance(model, x, y)

    # Conditional Feature Importance (classification:category, regression:quantile)
    index_groups = get_index_groups(model_type, y)
    conditional_features_importance = {
        str(label): feature_importance(model, x.iloc[index], y.iloc[index])
        for label, index in index_groups.items()
    }

    return PermutationFeatureImportance(
        model_type, model, x, y, features_importance, conditional_features_importance
    )
