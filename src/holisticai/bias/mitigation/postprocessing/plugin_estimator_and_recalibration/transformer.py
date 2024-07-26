from __future__ import annotations

import numpy as np
from holisticai.bias.mitigation.postprocessing.plugin_estimator_and_recalibration.algorithm import (
    PluginEstimationAndCalibrationAlgorithm,
)
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost


class PluginEstimationAndCalibration(BMPost):
    """
    Plugin Estimation and Calibration postprocessing optimizes over calibrated regressor outputs via a\
    smooth optimization. The rates of convergence of the proposed estimator were derived in terms of\
    the risk and fairness constraint.

    Parameters
    ----------
    L : int
        The number of thresholds to estimate.

    beta : float
        The regularization parameter

    References
    ----------
        .. [1] Chzhen, Evgenii, et al. "Fair regression via plug-in estimator and recalibration with statistical\
        guarantees." Advances in Neural Information Processing Systems 33 (2020): 19137-19148.
    """

    def __init__(self, L: int | None = 25, beta: float | None = 0.1):
        self.algorithm = PluginEstimationAndCalibrationAlgorithm(L=L, beta=beta)

    def fit(self, y_pred: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Compute a fair regression function by minimizing the squared error subject to a fairness constraint.

        Description
        ----------
        Compute a fair predictor by estimating a regression function by standard methods and then estimate\
        the thresholds to solve the minimization problem.

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (num_examples, ).
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
        Self
        """
        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]

        sensitive_features = np.stack([group_a, group_b], axis=1)
        self.algorithm.fit(y_pred, sensitive_features)
        return self

    def transform(
        self,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Apply transform function to predictions and likelihoods

        Description
        ----------
        Use a fitted probability to change the output label and invert the likelihood

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (nb_examples,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        threshold : float
            float value to discriminate between 0 and 1

        Returns
        -------
        dict
            A dictionary with new predictions
        """
        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]
        sensitive_features = np.stack([group_a, group_b], axis=1)

        new_y_pred = self.algorithm.transform(y_pred, sensitive_features)

        return {"y_pred": new_y_pred}

    def fit_transform(self, y_pred: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Fit and transform

        Description
        ----------
        Fit and transform

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (num_examples,).
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
        dict
            A dictionary with new predictions
        """
        return self.fit(
            y_pred,
            group_a,
            group_b,
        ).transform(y_pred, group_a, group_b)
