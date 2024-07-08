import numpy as np
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
        feature_importances = np.array(feature_importance.feature_importances.values[:, 1], dtype=float)
        if len(feature_importances) == 0 or sum(feature_importances) < tol:
            return 0 if self.divergence else 1

        importance = feature_importances
        feature_weight = importance / sum(importance)
        feature_equal_weight = np.array([1.0 / len(importance)] * len(importance))

        if self.divergence is True:
            return 1-jensenshannon(feature_weight, feature_equal_weight, base=2)
        else:  # noqa: RET505
            return entropy(feature_weight)/entropy(feature_equal_weight)


class SpreadDivergence(ImportanceSpread):
    name: str = "Spread Divergence"
    reference = 0
    divergence = True

def spread_divergence(feature_importance):
    metric = SpreadDivergence()
    return metric(feature_importance)

class SpreadRatio(ImportanceSpread):
    name: str = "Spread Ratio"
    reference = 0
    divergence = False

def spread_ratio(feature_importance):
    metric = SpreadRatio()
    return metric(feature_importance)
