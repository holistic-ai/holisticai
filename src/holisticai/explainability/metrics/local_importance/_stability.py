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
