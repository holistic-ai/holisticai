from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.utils import resample

from holisticai.utils.surrogate_models import Surrogate, get_features


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

    def __call__(self, X: Any, y_pred: Any, surrogate: Surrogate, num_bootstraps=5):
        features = get_features(surrogate)
        original_features = np.unique(features[features >= 0])
        similarities = []

        for i in range(num_bootstraps):
            X_resampled, y_resampled_pred = resample(X, y_pred, random_state=i, stratify=y_pred)  # type: ignore

            surrogate_resampled = deepcopy(surrogate)
            surrogate_resampled.fit(X_resampled, y_resampled_pred)

            resampled_features = np.unique(surrogate_resampled.feature[surrogate_resampled.feature >= 0])

            if len(original_features) > 0 and len(resampled_features) > 0:
                similarity = len(set(original_features).intersection(resampled_features)) / len(
                    set(original_features).union(resampled_features)
                )
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

    def __call__(self, X, y_pred, surrogate: Surrogate, num_bootstraps=5):
        original_importances = surrogate.feature_importances_
        weighted_similarities = []
        total_weight = 0

        for i in range(num_bootstraps):
            X_resampled, y_resampled_pred = resample(X, y_pred, random_state=i, stratify=y_pred)  # type: ignore

            surrogate_resampled = deepcopy(surrogate)
            surrogate_resampled.fit(X_resampled, y_resampled_pred)

            resampled_importances = getattr(surrogate_resampled, "feature_importances_", None)

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


def surrogate_features_stability(X, y_pred, surrogate, num_bootstraps=5):
    """
    Calculate the stability of features used in a surrogate model.
    The metric measures the similarity of features used in the surrogate model across different bootstraps.

    Parameters
    ----------
    X : Any
        The input data.

    y_pred : Any
        The predicted target values of the original model.

    surrogate : Surrogate
        The surrogate model.

    num_bootstraps : int
        The number of bootstraps to use.

    Returns
    -------
    float
        The stability of features used in a surrogate model.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.explainability.metrics.surrogate import (
    ...     surrogate_features_stability,
    ... )
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RandomForestClassifier()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> surrogate = RandomForestClassifier()
    >>> surrogate.fit(X, y_pred)
    >>> surrogate_features_stability(X, y_pred, surrogate)
    """
    m = FeaturesStability()
    return m(X, y_pred, surrogate, num_bootstraps)


def surrogate_feature_importances_stability(X, y_pred, surrogate, num_bootstraps=5):
    """
    Calculate the stability of features importances in a surrogate model.
    The metric measures the stability of feature importances across multiple bootstrap samples.

    Parameters
    ----------
    X : Any
        The input data.

    y_pred : Any
        The predicted target values of the original model.

    surrogate : Surrogate
        The surrogate model.

    num_bootstraps : int
        The number of bootstraps to use.

    Returns
    -------
    float
        The stability of features importances in a surrogate model.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.explainability.metrics.surrogate import (
    ...     surrogate_feature_importances_stability,
    ... )
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RandomForestClassifier()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> surrogate = RandomForestClassifier()
    >>> surrogate.fit(X, y_pred)
    >>> surrogate_feature_importances_stability(X, y_pred, surrogate)
    """
    m = FeatureImportancesStability()
    return m(X, y_pred, surrogate, num_bootstraps)
