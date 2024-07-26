import numpy as np
from holisticai.bias.mitigation.postprocessing.plugin_estimator_and_recalibration.algorithm_utils import f_lambda
from holisticai.utils.transformers.bias import SensitiveGroups


class PluginEstimationAndCalibrationAlgorithm:
    """Plugin Estimation and Calibration Algorithm

    This class implements the Plugin Estimation and Calibration Algorithm (PECA) for bias mitigation.
    This algorithm was proposed by Hardt et al. in their paper titled "Fair Regression via Plug-In Estimator and Recalibration".
    The paper can be found at: https://proceedings.neurips.cc/paper_files/paper/2020/file/ddd808772c035aed516d42ad3559be5f-Paper.pdf

    References
    ----------
    .. [1] Chzhen, E., Denis, C., Hebiri, M., Oneto, L., & Pontil, M. (2020).
        Fair regression via plug-in estimator and recalibration with statistical guarantees.
        Advances in Neural Information Processing Systems, 33, 19137-19148.
    """

    def __init__(self, L=25, beta=0.1):
        """Parameters
        ----------
        length : int, optional (default=25)
            The length of the range of values to consider for the output predictions.
            The range of values is [-length, length].
        beta : float, optional (default=0.1)
            The value of the beta parameter used in the calculation of the lambda values.

        Attributes
        ----------
        multiplier : int
            The multiplier used to convert the output predictions to the range [0, 1].
        length : int
            The length of the range of values to consider for the output predictions.
            The range of values is [-length, length].
        beta : float
            The value of the beta parameter used in the calculation of the lambda values.
        sensitive_groups : SensitiveGroups
            The SensitiveGroups object used to transform the sensitive features.
        epsilon : float
            The value of the epsilon parameter used to avoid division by zero.
        probabilities : list
            The probabilities for each sensitive group.
        lambda_values : np.ndarray
            The lambda values used in the calculation of the output predictions.
            self.multiplier = 1
            self.length = np.floor(L / 2).astype(np.int32)
            self.beta = beta
            self.sensitive_groups = SensitiveGroups()
            self.epsilon = np.finfo(float).eps
        """
        self.multiplier = 1
        self.length = np.floor(L / 2).astype(np.int32)
        self.beta = beta
        self.sensitive_groups = SensitiveGroups()
        self.epsilon = np.finfo(float).eps

    def fit(self, y_pred: np.ndarray, sensitive_features: np.ndarray):
        # Fit and transform the sensitive features
        transformed_sensitive_features = self.sensitive_groups.fit_transform(sensitive_features, convert_numeric=True)

        # Calculate the adjusted predictions
        adjusted_predictions = y_pred * 2 - 1

        # Calculate the probabilities for each sensitive group
        self.probabilities = np.array(
            [
                np.mean(transformed_sensitive_features == 0),
                np.mean(transformed_sensitive_features == 1),
            ]
        )

        # Compute the lambda values
        self.lambda_values = f_lambda(
            adjusted_predictions,
            transformed_sensitive_features,
            self.multiplier,
            self.length,
            self.beta,
        )

    def transform(self, y_pred: np.ndarray, sensitive_features: np.ndarray):
        transformed_sensitive_features = self.sensitive_groups.transform(sensitive_features, convert_numeric=True)
        adjusted_predictions = y_pred * 2 - 1
        index_range = np.arange(-self.length, self.length + 1, 1)

        transformed_sensitive_features = transformed_sensitive_features.to_list()
        minimizing_values = (
            np.expand_dims(self.probabilities[transformed_sensitive_features], axis=1)
            * np.square(np.expand_dims(adjusted_predictions, axis=1) - index_range * self.multiplier / self.length)
            + (1 - 2 * np.expand_dims(transformed_sensitive_features, axis=1)) * self.lambda_values
        )
        min_indices = np.argmin(minimizing_values, axis=1)
        output_predictions = index_range[min_indices] * self.multiplier / self.length
        output_predictions = (output_predictions + 1) / 2
        return output_predictions
