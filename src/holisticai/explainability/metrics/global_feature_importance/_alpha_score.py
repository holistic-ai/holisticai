from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holisticai.typing._typing import ArrayLike

from holisticai.utils._commons import get_top_ranking_from_scores


class AlphaScore:
    """
    Represents the Fourth Fifths metric.
    """

    reference: int = 0
    name: str = "Alpha Importance Score"

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def __call__(self, feature_importances: ArrayLike):
        """
        Calculates the alpha importance from feature importance values, identifying the smallest \
        proportion of features that collectively represent at least alpha percent of the total feature importance.

        Parameters
        ----------
        feature_importance: ArrayLike
            The feature importance values.

        Returns:
            float: The alpha importance value.
        """
        return get_top_ranking_from_scores(feature_importances) / len(feature_importances)


def alpha_score(feature_importance: ArrayLike, alpha: float = 0.8):
    """
    Alpha Score calculates the smallest proportion of features that account for the alpha percentage of the overall feature importance.

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
