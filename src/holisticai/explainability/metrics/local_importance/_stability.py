import numpy as np
from holisticai.explainability.commons import LocalImportances


class DataStability:
    reference: int = 0
    name: str = "Data Stability"

    def __call__(self, local_feature_importance):
        feature_importances = local_feature_importance.values
        # Calculate the median for each instance (row)
        medians_row = np.median(feature_importances, axis=1)

        # Calculate the interquartile range (IQR) for each instance
        q75_row, q25_row = np.percentile(feature_importances, [75, 25], axis=1)
        iqr_row = q75_row - q25_row

        # Calculate the Normalized Interquartile Range (nIQR) for each instance
        niqr_row = iqr_row / medians_row

        # Calculate the mean of the nIQR to obtain a global measure of Data Stability
        return np.mean(niqr_row)


def data_stability(local_feature_importance: LocalImportances):
    """
    Calculate the data stability metric for local feature importances.

    This function computes the data stability metric, which measures the consistency
    of feature importances across different instances in the dataset. The data stability
    metric is calculated as the mean normalized interquartile range (nIQR) of the feature
    importances. A higher data stability score indicates a higher level of consistency in
    the feature importances across instances.

    The data stability metric is calculated using the DataStability class, which calculates
    the normalized interquartile range (nIQR) of the feature importances. The nIQR is a
    measure of the spread of the feature importances and provides a global measure of stability.

    Parameters
    ----------
    local_feature_importance: LocalImportances
      A LocalImportances object containing the feature importances for each instance in the dataset.

    Returns
    -------
    float
      The mean normalized interquartile range (nIQR) of the feature importances, serving as the data stability metric.

    Examples
    --------
    >>> import pandas as pd
    >>> from holisticai.explainability.commons import LocalImportances
    >>> importances = pd.DataFrame({
    ...     "feature_1": [0.10, 0.20, 0.30],
    ...     "feature_2": [0.10, 0.25, 0.35],
    ...     "feature_3": [0.15, 0.20, 0.30]
    ... })
    >>> local_importances = LocalImportances(importances)
    >>> stability_score = data_stability(local_importances)
    >>> print(stability_score)
    """
    metric = DataStability()
    return metric(local_feature_importance)


class FeatureStability:
    reference: int = 0
    name: str = "Feature Stability"

    def __call__(self, local_feature_importance: LocalImportances):
        feature_importances = local_feature_importance.values
        # Calculate the median for each feature (column)
        medians = np.median(feature_importances, axis=0)

        # Calculate the interquartile range (IQR) for each feature
        q75, q25 = np.percentile(feature_importances, [75, 25], axis=0)
        iqr = q75 - q25

        # Calculate the Normalized Interquartile Range (nIQR) for each feature
        niqr = iqr / medians

        # Calculate the mean of the nIQR to obtain a global measure of Feature Stability
        return np.mean(niqr)


def feature_stability(local_feature_importance: LocalImportances):
    """
    Calculate the feature stability metric for local feature importances.

    This function computes the feature stability metric, which measures the consistency
    of feature importances for individual features across different instances in the dataset.
    The feature stability metric is calculated as the mean normalized interquartile range (nIQR)
    of the feature importances. A higher feature stability score indicates a higher level of
    consistency in the importance of each feature across instances.

    The feature stability metric is calculated using the FeatureStability class, which calculates
    the normalized interquartile range (nIQR) of the feature importances. The nIQR is a measure
    of the spread of the feature importances and provides a global measure of stability for each feature.

    Parameters
    ----------
    local_feature_importance: LocalImportances
      A LocalImportances object containing the feature importances for each instance in the dataset.

    Returns
    -------
    float
      The mean normalized interquartile range (nIQR) of the feature importances, serving as the feature stability metric.

    Examples
    --------
    >>> import pandas as pd
    >>> from holisticai.explainability.commons import LocalImportances
    >>> importances = pd.DataFrame({
    ...     "feature_1": [0.10, 0.20, 0.30],
    ...     "feature_2": [0.10, 0.25, 0.35],
    ...     "feature_3": [0.15, 0.20, 0.30]
    ... })
    >>> local_importances = LocalImportances(importances)
    >>> stability_score = feature_stability(local_importances)
    >>> print(stability_score)
    """
    metric = FeatureStability()
    return metric(local_feature_importance)
