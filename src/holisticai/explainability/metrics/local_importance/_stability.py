import numpy as np
from holisticai.explainability.metrics.global_importance._importance_spread import spread_divergence
from holisticai.utils import Importances, LocalImportances


class DataStability:
    reference: int = 1
    name: str = "Data Stability"

    def __call__(self, local_feature_importance: LocalImportances):
        spreads = []
        for weights in np.array(local_feature_importance.values):
            from holisticai.utils import Importances

            spread = spread_divergence(
                Importances(values=weights, feature_names=local_feature_importance.feature_names)
            )
            spreads.append(spread)
        instances_names = list(range(local_feature_importance.values.shape[0]))
        data_divergence = Importances(values=spreads, feature_names=instances_names)
        return spread_divergence(data_divergence)


def data_stability(local_feature_importance: LocalImportances):
    """
    Calculate the data stability score based on the local feature importances.
    Data Stability measures the stability of the feature importance values across different instances.
    Higher values indicate more stable feature importance values across instances.

    Parameters
    ----------
    local_feature_importance : LocalImportances
      The local feature importances calculated for each instance.

    Returns
    -------
    float
      The data stability score.

    Examples
    --------
    >>> import pandas as pd
    >>> from holisticai.explainability.commons import LocalImportances
    >>> importances = pd.DataFrame(
    ...     {
    ...         "feature_1": [0.10, 0.20, 0.30],
    ...         "feature_2": [0.10, 0.25, 0.35],
    ...         "feature_3": [0.15, 0.20, 0.30],
    ...     }
    ... )
    >>> local_importances = LocalImportances(importances)
    >>> stability_score = data_stability(local_importances)
    >>> print(stability_score)
    """
    ds = DataStability()
    return ds(local_feature_importance)


class FeatureStability:
    reference: int = 1
    name: str = "Feature Stability"

    def __call__(self, local_feature_importance: LocalImportances):
        transposed_data = np.array(local_feature_importance.values).T
        transposed_data_norm = transposed_data / np.sum(transposed_data, axis=0)
        spreads = []
        instances_names = list(range(local_feature_importance.values.shape[0]))
        for weights in transposed_data_norm:
            spread = spread_divergence(Importances(values=weights, feature_names=instances_names))
            spreads.append(spread)

        feature_divergence = Importances(values=spreads, feature_names=local_feature_importance.feature_names)
        return spread_divergence(feature_divergence)


def feature_stability(local_feature_importance: LocalImportances):
    """
    Calculate the feature stability score based on the local feature importances.
    Feature Stability measures the stability of the feature importance values across different features.
    Higher values indicate more stable feature importance values across features.

    Parameters
    ----------
    local_feature_importance : LocalImportances
      The local feature importances calculated for each instance.

    Returns
    -------
    float
      The feature stability score.

    Examples
    --------
    >>> import pandas as pd
    >>> from holisticai.explainability.commons import LocalImportances
    >>> importances = pd.DataFrame(
    ...     {
    ...         "feature_1": [0.10, 0.20, 0.30],
    ...         "feature_2": [0.10, 0.25, 0.35],
    ...         "feature_3": [0.15, 0.20, 0.30],
    ...     }
    ... )
    >>> local_importances = LocalImportances(importances)
    >>> stability_score = feature_stability(local_importances)
    >>> print(stability_score)
    """
    fs = FeatureStability()
    return fs(local_feature_importance)
