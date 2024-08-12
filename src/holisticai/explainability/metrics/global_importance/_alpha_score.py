from __future__ import annotations

from holisticai.utils import Importances


def calculate_alpha_importance(feature_importance: Importances, alpha=0.8):
    """
    Calculates the alpha importance of a feature based on its feature importance.

    Parameters
    ----------
    feature_importance: list
        A list of feature importance values.
    alpha: (float, optional)
        The threshold value for filtering feature importance. Defaults to 0.8.

    Returns
    -------
        float: The alpha importance of the feature, calculated as the ratio of the number of features with importance
               greater than or equal to alpha to the total number of features.

    """
    alpha_feat_imp = feature_importance.top_alpha(alpha)
    len_alpha = len(alpha_feat_imp)
    len_100 = len(feature_importance)
    return len_alpha / len_100


class AlphaScore:
    """
    Represents the Fourth Fifths metric.
    """

    reference: int = 0
    name: str = "Alpha Importance Score"

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def __call__(self, feature_importances: Importances):
        """
        Calculates the alpha importance of feature importance values.

        Parameters
        ----------
        feature_importance: Importances
            The feature importance values.

        Returns:
            float: The alpha importance value.
        """
        ranked_feature_importances = feature_importances.top_alpha(alpha=self.alpha)
        len_alpha = len(ranked_feature_importances)
        len_100 = len(feature_importances)
        return len_alpha / len_100


def alpha_score(feature_importance: Importances, alpha: float = 0.8):
    """
    Alpha Score calculates the proportion of features that account for the alpha percentage of the overall feature importance.

    Parameters
    ----------
    feature_importance: Importances
        The feature importance values.

    alpha: float, optional
        The alpha value represents the percentage of importance that will be considered in the calculation.
        For example, if alpha=0.8, the top 80% of the most important features will be considered.

    Returns
    -------
    float
        The alpha importance score

    Examples
    --------
    >>> from holisticai.explainability.commons import Importance
    >>> from holisticai.explainability.metrics import alpha_score
    >>> values = np.array([0.50, 0.30, 0.20])
    >>> feature_names = ["feature_1", "feature_2", "feature_3"]
    >>> feature_importance = Importance(values=values, feature_names=feature_names)
    >>> alpha_score(feature_importance)
    0.6666666666666666
    """

    metric = AlphaScore(alpha=alpha)
    return metric(feature_importance)
