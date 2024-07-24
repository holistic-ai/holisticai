import numpy as np
from holisticai.utils import Importances
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


class ImportanceSpread:
    def __call__(self, feature_importance):
        """
        Parameters
        ----------
        feature_importance: np.array
            array with raw feature importance
        divergence: bool
            if True calculate the inverse Jensen-Shannon divergence, otherwise the ratio
        """
        tol = 1e-8
        feature_importances = np.array(feature_importance.values, dtype=float)
        if len(feature_importances) == 0 or sum(feature_importances) < tol:
            return 0 if self.divergence else 1

        importance = feature_importances
        feature_weight = importance / sum(importance)
        feature_equal_weight = np.array([1.0 / len(importance)] * len(importance))

        if self.divergence:
            metric = 1 - jensenshannon(feature_weight, feature_equal_weight, base=2)
        else:
            metric = entropy(feature_weight) / entropy(feature_equal_weight)
        return float(metric)


class SpreadRatio(ImportanceSpread):
    name: str = "Spread Ratio"
    reference: float = 0
    divergence: bool = False


class SpreadDivergence(ImportanceSpread):
    name: str = "Spread Divergence"
    reference: float = 0
    divergence: bool = True


def spread_ratio(feature_importance: Importances):
    """
    Calculates the spread ratio of the given feature importance.
    The spread ratio measures the degree of evenness or concentration in the distribution of feature importance values.
    A higher spread ratio indicates a more evenly distributed feature importance, while a lower spread ratio indicates a more concentrated feature importance.
    A lower ratio concentrates the importances and facilitates interpretability.

    Parameters
    ----------
    feature_importance: Importances
        The feature importance values for the features.

    Returns
    -------
    float:
        The spread ratio of the feature importance.

    Examples
    --------

    >>> from holisticai.explainability.commons import Importances
    >>> from holisticai.explainability.metrics.global_importance import spread_ratio
    >>> values = np.array([0.10, 0.20, 0.30])
    >>> feature_names = ["feature_1", "feature_2", "feature_3"]
    >>> feature_importance = Importances(values=values, feature_names=feature_names)
    >>> score = spread_ratio(feature_importance)
    0.9206198357143052

    """
    metric = SpreadRatio()
    return metric(feature_importance)


def spread_divergence(feature_importance: Importances):
    """
    Calculates the spread divergence metric based on the inverse of the Jensen-Shannon distance
    (square root of the Jensen-Shannon divergence), for a given feature importance.
    A lower ratio concentrates the importances and facilitates interpretability.

    Parameters
    ----------
    feature_importance: Importances
        The feature importance values for the features.

    Returns
    -------
        float: The spread divergence metric value.

    Example
    -------

    >>> from holisticai.explainability.commons import Importances
    >>> from holisticai.explainability.metrics.global_importance import (
    ...     spread_divergence,
    ... )
    >>> feature_importance = Importances(
    ...     values=np.array([0.10, 0.20, 0.30]),
    ...     feature_names=["feature_1", "feature_2", "feature_3"],
    ... )
    >>> score = spread_divergence(feature_importance)
    0.8196393599933761

    """
    metric = SpreadDivergence()
    return metric(feature_importance)
