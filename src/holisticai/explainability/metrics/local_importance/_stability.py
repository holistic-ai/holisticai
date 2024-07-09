import numpy as np


class DataStability:
    reference: int = 0
    name: str = "Data Stability"

    def __call__(self, local_feature_importance):
        feature_importances = local_feature_importance.feature_importances.values
        # Calculate the median for each instance (row)
        medians_row = np.median(feature_importances, axis=1)

        # Calculate the interquartile range (IQR) for each instance
        q75_row, q25_row = np.percentile(feature_importances, [75, 25], axis=1)
        iqr_row = q75_row - q25_row

        # Calculate the Normalized Interquartile Range (nIQR) for each instance
        niqr_row = iqr_row / medians_row

        # Calculate the mean of the nIQR to obtain a global measure of Data Stability
        return np.mean(niqr_row)


def data_stability(local_feature_importance):
    """
    Calculate the data stability metric for local feature importances.

    This function computes the data stability metric, which measures the consistency
    of feature importances across different instances in the dataset. It leverages the
    DataStability class to calculate the normalized interquartile range (nIQR) of the
    feature importances, providing a global measure of stability.

    Parameters:
    - local_feature_importance (array-like): A 2D array or list where each row represents
      the feature importances for a single instance in the dataset.

    Returns:
    - float: The mean normalized interquartile range (nIQR) of the feature importances,
      serving as the data stability metric.

    Example:
    >>> local_importances = [[0.1, 0.2, 0.3], [0.1, 0.25, 0.35], [0.15, 0.2, 0.3]]
    >>> stability_score = data_stability(local_importances)
    >>> print(stability_score)
    """
    metric = DataStability()
    return metric(local_feature_importance)


class FeatureStability:
    reference: int = 0
    name: str = "Feature Stability"

    def __call__(self, local_feature_importance):
        feature_importances = local_feature_importance.feature_importances.values
        # Calculate the median for each feature (column)
        medians = np.median(feature_importances, axis=0)

        # Calculate the interquartile range (IQR) for each feature
        q75, q25 = np.percentile(feature_importances, [75, 25], axis=0)
        iqr = q75 - q25

        # Calculate the Normalized Interquartile Range (nIQR) for each feature
        niqr = iqr / medians

        # Calculate the mean of the nIQR to obtain a global measure of Feature Stability
        return np.mean(niqr)


def feature_stability(local_feature_importance):
    """
    Calculate the feature stability metric for local feature importances.

    This function computes the feature stability metric, which assesses the consistency
    of feature importances for individual features across different instances in the dataset.
    It utilizes the FeatureStability class to calculate a stability score for each feature,
    reflecting how stable the importance of each feature is across the dataset.

    Parameters:
    - local_feature_importance (array-like): A 2D array or list where each row represents
      the feature importances for a single instance in the dataset.

    Returns:
    - dict: A dictionary where each key is a feature index and each value is the stability
      score of that feature, indicating the consistency of its importance across instances.

    Example:
    >>> local_importances = [[0.1, 0.2, 0.3], [0.1, 0.25, 0.35], [0.15, 0.2, 0.3]]
    >>> stability_scores = feature_stability(local_importances)
    >>> print(stability_scores)
    """
    metric = FeatureStability()
    return metric(local_feature_importance)
