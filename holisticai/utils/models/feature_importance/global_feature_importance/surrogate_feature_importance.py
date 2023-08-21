"""
This module provides functions for computing surrogate feature importance and efficacy metrics.
"""

from holisticai.explainability.feature_importance import SurrogateFeatureImportance
from holisticai.utils._validation import _matrix_like_to_dataframe, _array_like_to_series
import pandas as pd


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
    if model_type == 'binary_classification':
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=3)
        return dt.fit(x, y)
    elif model_type == 'regression':
        from sklearn.tree import DecisionTreeRegressor
        dt = DecisionTreeRegressor(max_depth=3)
        return dt.fit(x, y)
    else:
        raise ValueError("model_type must be either 'binary_classification' or 'regression'")

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
    
    sorted_features = sorted(zip(feature_names, forest.feature_importances_), key=lambda x: x[1], reverse=True)
    features_dict = dict(sorted_features)

    features_importance = pd.DataFrame({'Variable':features_dict.keys(), 'Importance':features_dict.values()})
    features_importance["Importance"] = abs(features_importance["Importance"])
    return SurrogateFeatureImportance(model_type, model, x, y, features_importance, surrogate)