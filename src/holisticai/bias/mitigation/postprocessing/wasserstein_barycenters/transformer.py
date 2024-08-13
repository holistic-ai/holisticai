import numpy as np
from holisticai.bias.mitigation.postprocessing.wasserstein_barycenters.algorithm import WassersteinBarycenterAlgorithm
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost


class WassersteinBarycenter(BMPost):
    """
    Fair Regression with Wasserstein Barycenters learning a real-valued function that\
    satisfies the Demographic Parity constraint [1]_ . The strategy founds the optimal fair\
    predictor computing the Wasserstein barycenter of the distributions induced by the\
    standard regression function on the sensitive groups.

    References
    ----------
        .. [1] Chzhen, Evgenii, et al. "Fair regression with wasserstein barycenters."\
        Advances in Neural Information Processing Systems 33 (2020): 7321-7331.
    """

    def __init__(self):
        self.algorithm = WassersteinBarycenterAlgorithm()

    def fit(self, y_pred: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Compute parameters for calibrated equalized odds.

        Description
        ----------
        Compute parameters for calibrated equalized odds algorithm.

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
