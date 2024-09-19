from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.base import clone
from sklearn.utils import resample

from holisticai.utils._definitions import ModelProxy
from holisticai.utils.models.surrogate import Surrogate, get_feature_importances, get_features


class FeaturesStability:
    """
    FeaturesStability calculates the stability of features used in a surrogate model.
    The metric measures the similarity of features used in the surrogate model across different bootstraps.

    Parameters
    ----------
        reference (float): The reference of best stability value = 1.
        name (str): The name of the stability metric: "Features Stability".
    """
    reference: float = 1
    name: str = "Features Stability"


    def __call__(self, proxy: ModelProxy, surrogate: Surrogate, X: Any, stratify=None, num_bootstraps=5):
        features = get_features(surrogate)
        original_features = np.unique(features[features >= 0])
        similarities = []

        for i in range(num_bootstraps):
            X_resampled = resample(X, random_state=i, stratify=stratify)
            y_resampled_pred = proxy.predict(X_resampled)

            clf_resampled = clone(surrogate._surrogate)  # noqa: SLF001
            clf_resampled.fit(X_resampled, y_resampled_pred)

            resampled_features = np.unique(clf_resampled.tree_.feature[clf_resampled.tree_.feature >= 0])

            if len(original_features) > 0 and len(resampled_features) > 0:
                similarity = len(set(original_features).intersection(resampled_features)) / len(set(original_features).union(resampled_features))
                similarities.append(similarity)

        if len(similarities) == 0:
            return 0.0

        S_features = np.mean(similarities)
        return float(S_features)


class FeatureImportancesStability:
    """
    FeaturesStability calculates the stability of features importances in a surrogate model.
    The metric measures the stability of feature importances across multiple bootstrap samples.

    Parameters
    ----------
        reference (float): The reference of best stability value = 1.
        name (str): The name of the stability metric: "Features Importances Stability".
    """
    reference: float = 1
    name: str = "Feature Importances Stability"


    def __call__(self, proxy, surrogate, X, stratify=None, num_bootstraps=5):

        original_importances = get_feature_importances(surrogate)
        weighted_similarities = []
        total_weight = 0

        for i in range(num_bootstraps):
            X_resampled = resample(X, random_state=i, stratify=stratify)
            y_resampled_pred = proxy.predict(X_resampled)

            clf_resampled = clone(surrogate._surrogate)  # noqa: SLF001
            clf_resampled.fit(X_resampled, y_resampled_pred)

            resampled_importances = getattr(clf_resampled, 'feature_importances_', None)

            if resampled_importances is None or len(resampled_importances) == 0:
                resampled_importances = np.zeros_like(original_importances)

            norm_original = np.linalg.norm(original_importances)
            norm_resampled = np.linalg.norm(resampled_importances)

            if norm_original != 0 and norm_resampled != 0:
                similarity = np.dot(original_importances, resampled_importances) / (norm_original * norm_resampled)
                weight = (norm_original + norm_resampled) / 2  # Ponderar por la magnitud de las normas
                weighted_similarities.append(similarity * weight)
                total_weight += weight

        if total_weight == 0:
            return 0.0

        S_importances = np.sum(weighted_similarities) / total_weight
        return float(S_importances)


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

    def __call__(self, surrogate):
        tol = 1e-8
        feature_importance_values = np.array(surrogate._surrogate.feature_importances_, dtype=float)  # noqa: SLF001
        if len(feature_importance_values) == 0 or sum(feature_importance_values) < tol:
            return 0

        feature_weight = feature_importance_values / sum(feature_importance_values)
        feature_equal_weight = np.array([1.0 / len(feature_importance_values)] * len(feature_importance_values))

        metric = 1 - jensenshannon(feature_weight, feature_equal_weight, base=2)
        return float(metric)


def features_stability(proxy, surrogate, X, stratify=None, num_bootstraps=5):
    m = FeaturesStability()
    return m(proxy, surrogate, X, stratify, num_bootstraps)

def feature_importances_stability(proxy, surrogate, X, stratify=None, num_bootstraps=5):
    m = FeatureImportancesStability()
    return m(proxy, surrogate, X, stratify, num_bootstraps)

def feature_importances_spread(surrogate):
    m = FeatureImportanceSpread()
    return m(surrogate)
