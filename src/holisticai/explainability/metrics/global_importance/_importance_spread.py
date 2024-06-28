import numpy as np


class ImportanceSpread:
    def __call__(self, feature_importance):
        """
        Parameters
        ----------
        feature_importance: np.array
            array with raw feature importance
        divergence: bool
            if True calculate divergence instead of ratio
        """
        feature_importances = np.array(feature_importance.feature_importances.values[:,1], dtype=float)
        if len(feature_importances) == 0 or sum(feature_importances) < 1e-8:
            return 0 if self.divergence else 1

        importance = feature_importances
        from scipy.stats import entropy

        feature_weight = importance / sum(importance)
        feature_equal_weight = np.array([1.0 / len(importance)] * len(importance))

        # entropy or divergence
        if self.divergence is False:
            return entropy(feature_weight) / entropy(feature_equal_weight)  # ratio
        else:  # noqa: RET505
            return entropy(feature_weight, feature_equal_weight)  # divergence


class SpreadDivergence(ImportanceSpread):
    name : str = "Spread Divergence"
    reference = np.inf
    divergence = True

class SpreadRatio(ImportanceSpread):
    name : str = "Spread Ratio"
    reference = 0
    divergence = False

def spread_ratio(feature_importance):
    metric = SpreadRatio()
    return metric(feature_importance)