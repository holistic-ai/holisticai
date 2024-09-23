import numpy as np
from holisticai.typing import ArrayLike
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


class FeatureImportanceSpread:
    """
    FeatureImportanceSpread meausre the concentration of feature importances in a surrogate model.

    Parameters
    ----------

    name : str
        The name of the metric, which is "Spread Divergence".
    reference : float
        The reference value for the metric, initialized to 0.
    """

    name: str = "Spread Divergence"
    reference: float = 0
    divergence: bool = True

    def __call__(self, feature_importances: ArrayLike):
        tol = 1e-8
        feature_importance_values = np.array(feature_importances, dtype=float)
        if len(feature_importance_values) == 0 or sum(feature_importance_values) < tol:
            return 0

        if len(feature_importance_values) == 1:
            return 1.0

        feature_weight = feature_importance_values / sum(feature_importance_values)
        feature_equal_weight = np.array([1.0 / len(feature_importance_values)] * len(feature_importance_values))

        if self.divergence:
            metric = 1 - jensenshannon(feature_weight, feature_equal_weight, base=2)
        else:
            metric = entropy(feature_weight) / entropy(feature_equal_weight)
        return float(metric)


def feature_importances_spread(feature_importances: ArrayLike):
    m = FeatureImportanceSpread()
    return m(feature_importances)


class SpreadRatio(FeatureImportanceSpread):
    name: str = "Spread Ratio"
    reference: float = 0
    divergence: bool = False


class SpreadDivergence(FeatureImportanceSpread):
    name: str = "Spread Divergence"
    reference: float = 0
    divergence: bool = True


def spread_ratio(feature_importance: ArrayLike):
    """
    The spread ratio, ranging from 0 to 1, measures the degree of evenness or concentration in the distribution of feature importance values.
    A higher spread ratio indicates a more evenly distributed feature importance, while a lower spread ratio indicates a more concentrated feature importance.
    A lower ratio concentrates the importances and facilitates interpretability.

    Parameters
    ----------
    feature_importance: ArrayLike
        The feature importance values for the features.

    Returns
    -------
    float:
        The spread ratio of the feature importance.

    Examples
    --------

    >>> from holisticai.explainability.metrics.global_feature_importance import (
    ...     spread_ratio,
    ... )
    >>> feature_importance = np.array([0.10, 0.20, 0.30])
    >>> score = spread_ratio(feature_importance)
    0.9206198357143052

    """
    metric = SpreadRatio()
    return metric(feature_importance)


def spread_divergence(feature_importance: ArrayLike):
    """
    Calculates the spread divergence metric based on the inverse of the Jensen-Shannon distance
    (square root of the Jensen-Shannon divergence), for a given feature importance.
    A lower ratio concentrates the importances and facilitates interpretability.

    Parameters
    ----------
    feature_importance: ArrayLike
        The feature importance values for the features.

    Returns
    -------
        float: The spread divergence metric value.

    Example
    -------

    >>> from holisticai.explainability.metrics import spread_divergence
    >>> feature_importance = np.array([0.10, 0.20, 0.30])
    >>> score = spread_divergence(feature_importance)
    0.8196393599933761

    """
    metric = SpreadDivergence()
    return metric(feature_importance)
