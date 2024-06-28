from __future__ import annotations

from holisticai.explainability.commons._feature_importance import filter_feature_importance
from pydantic import BaseModel


def calculate_alpha_importance(feature_importance, alpha=0.8):
    """
    Calculates the alpha importance of a feature based on its feature importance.

    Parameters:
        feature_importance (list): A list of feature importance values.
        alpha (float, optional): The threshold value for filtering feature importance. Defaults to 0.8.

    Returns:
        float: The alpha importance of the feature, calculated as the ratio of the number of features with importance
               greater than or equal to alpha to the total number of features.

    """
    alpha_feat_imp = filter_feature_importance(feature_importance, alpha)
    len_alpha = len(alpha_feat_imp)
    len_100 = len(feature_importance)
    return len_alpha / len_100


class AlphaImportanceScore(BaseModel):
    """
    Represents the Fourth Fifths metric.

    Attributes:
        reference (int): The reference value.
        name (str): The name of the metric.
        alpha (float): The alpha value.

    """

    reference: int = 0
    name: str = "Alpha Importance Score"
    alpha: float = 0.8

    def __call__(self, feature_importances: list[float], ranked_feature_importances: list[float]):
        """
        Calculates the alpha importance of feature importance values.

        Args:
            feature_importance (list[float]): The feature importance values.
            cond_feat_imp (list[list[float]]): The conditional feature importance values.
            detailed (bool, optional): Whether to return detailed alpha importance values for each conditional feature importance. Defaults to False.

        Returns:
            float or tuple: The alpha importance value or a tuple of alpha importance values and detailed alpha importance values.
        """
        len_alpha = len(ranked_feature_importances)
        len_100 = len(feature_importances)
        return len_alpha / len_100


def alpha_importance_score(feature_importance, ranked_feature_importance):
    metric = AlphaImportanceScore()
    return metric(feature_importance, ranked_feature_importance)
