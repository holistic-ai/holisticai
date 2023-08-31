import numpy as np
import pandas as pd

from holisticai.explainability.feature_importance import LimeFeatureImportance
from holisticai.explainability.metrics.utils import get_index_groups, lime_creator
from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)


def compute_lime_feature_importance(model_type, model, x, y):
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    if not isinstance(y, pd.Series):
        y = _array_like_to_series(y)

    if model_type == "binary_classification":
        lime_mode = "classification"
        scorer = model.predict_proba

    elif model_type == "regression":
        lime_mode = "regression"
        scorer = model.predict

    index_groups = get_index_groups(model_type, y)
    features_importance = lime_creator(
        scorer=scorer, X=x, index_groups=index_groups, mode=lime_mode
    )
    conditional_features_importance = {
        str(c): gdf for c, gdf in features_importance.groupby("Sample Group")
    }

    return LimeFeatureImportance(features_importance, conditional_features_importance)
