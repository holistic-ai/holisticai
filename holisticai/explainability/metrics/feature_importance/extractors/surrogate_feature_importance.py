"""
This module provides functions for computing surrogate feature importance and efficacy metrics.
"""

import dtreeviz
import pandas as pd

from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)

from ..global_importance import (
    fourth_fifths,
    global_explainability_ease_score,
    importance_spread_divergence,
    importance_spread_ratio,
    surrogate_efficacy,
)

from holisticai.explainability.plots import DecisionTreeVisualizer
from .extractor_utils import BaseFeatureImportance, GlobalFeatureImportance, get_top_k


def create_surrogate_model(model_type, x, y):
    """
    Create a surrogate model for a given model type, input features and predicted output.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        x (pandas.DataFrame): The input features.
        y (numpy.ndarray): The predicted output.

    Returns:
        sklearn.tree.DecisionTreeClassifier or sklearn.tree.DecisionTreeRegressor: The surrogate model.
    """
    if model_type == "binary_classification":
        from sklearn.tree import DecisionTreeClassifier

        dt = DecisionTreeClassifier(max_depth=3)
        return dt.fit(x, y)
    elif model_type == "regression":
        from sklearn.tree import DecisionTreeRegressor

        dt = DecisionTreeRegressor(max_depth=3)
        return dt.fit(x, y)
    else:
        raise ValueError(
            "model_type must be either 'binary_classification' or 'regression'"
        )


def compute_surrogate_feature_importance(model_type, model, x, y):
    """
    Compute surrogate feature importance for a given model type, model and input features.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        model (sklearn estimator): The model to compute surrogate feature importance for.
        x (pandas.DataFrame): The input features.

    Returns:
        holisticai.explainability.feature_importance.SurrogateFeatureImportance: The surrogate feature importance.
    """
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    if not isinstance(y, pd.DataFrame):
        y = _array_like_to_series(y)

    y_pred = model.predict(x)
    surrogate = create_surrogate_model(model_type, x, y_pred)
    feature_names = x.columns
    forest = surrogate

    sorted_features = sorted(
        zip(feature_names, forest.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    features_dict = dict(sorted_features)
    df_feat_imp = pd.DataFrame(
        {"Variable": features_dict.keys(), "Importance": features_dict.values()}
    )

    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp["Importance"] /= df_feat_imp["Importance"].sum()
    df_feat_imp = df_feat_imp.sort_values("Importance", ascending=False).copy()

    return SurrogateFeatureImportance(model_type, model, x, y, df_feat_imp, surrogate)


class SurrogateFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(self, model_type, model, x, y, importance_weights, surrogate):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.feature_importance = importance_weights
        self.surrogate = surrogate
        self.tree_visualizer = DecisionTreeVisualizer()

    def get_topk(self, top_k):
        if top_k is None:
            feat_imp = self.feature_importance
        else:
            feat_imp = get_top_k(self.feature_importance, top_k)

        return {"feature_importance": feat_imp}

    def metrics(self, feature_importance, detailed):

        reference_values = {
            "Fourth Fifths": 0,
            "Importance Spread Divergence": "-",
            "Importance Spread Ratio": 0,
            "Global Explainability Ease Score": 1,
            "Surrogate Efficacy Classification": 1,
            "Surrogate Efficacy Regression": 0,
        }

        metrics = pd.concat(
            [
                fourth_fifths(feature_importance),
                importance_spread_divergence(feature_importance),
                importance_spread_ratio(feature_importance),
                global_explainability_ease_score(
                    self.model_type, self.model, self.x, self.y, feature_importance
                ),
                surrogate_efficacy(self.model_type, self.x, self.y, self.surrogate),
            ],
            axis=0,
        )

        reference_column = pd.DataFrame(
            [reference_values.get(metric) for metric in metrics.index],
            columns=["Reference"],
        ).set_index(metrics.index)
        metrics_with_reference = pd.concat([metrics, reference_column], axis=1)

        return metrics_with_reference

    def tree_visualization(self, backend='sklearn'):
        if backend in self.tree_visualizer.visualization_backend:
            return self.tree_visualizer.show(backend, self)    
        else:
            raise("Unknown backend")
        