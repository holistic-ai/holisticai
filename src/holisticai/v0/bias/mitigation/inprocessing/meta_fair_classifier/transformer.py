from typing import Optional

import numpy as np

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from .algorithm import MetaFairClassifierAlgorithm
from .algorithm_utils import MFLogger
from .constraints import FalseDiscovery, StatisticalRate


class MetaFairClassifier(BMImp):
    """
    The meta algorithm here takes the fairness metric as part of the input
    and returns a classifier optimized w.r.t. that fairness metric.
    The algorithm support only binary protected groups -> group_a = 1 - group_b

    References
    ----------
        Celis, L. Elisa, et al. "Classification with fairness constraints:
        A meta-algorithm with provable guarantees." Proceedings of the conference on
        fairness, accountability, and transparency. 2019.
    """

    def __init__(
        self,
        tau: Optional[float] = 0.8,
        constraint: Optional[str] = "StatisticalRate",
        seed: Optional[int] = None,
        verbose: Optional[int] = 0,
    ):
        """
        Parameters
        ----------
            tau : float
                Fairness penalty parameter (0,1). Higher parameter increate the threshold
                for a valid estimator.

            constraint : str
                The type of fairness metric to be used, currently supported:
                - StatisticalRate: Statistical rate/disparate impact
                - FalseDiscovery: False discovery rate ratio

            seed : int
                Random seed.
        """
        # Constant parameters
        steps = 10
        eps = 0.01
        mu = 0.01

        logger = MFLogger(tau, eps=eps, steps=steps, verbose=verbose)

        if constraint == "StatisticalRate":
            constraint = StatisticalRate(mu=mu)

        elif constraint == "FalseDiscovery":
            constraint = FalseDiscovery(mu=mu)

        self.algorithm = MetaFairClassifierAlgorithm(
            constraint=constraint, tau=tau, eps=eps, steps=steps, logger=logger
        )
        self.seed = seed

    def transform_estimator(self):
        return self

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit the model

        Description
        -----------
        Learn a fair classifier.

        Parameters
        ----------

        X : numpy array
            input matrix

        y_true : numpy array
            target vector

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector

        Returns
        -------
        the same object
        """
        params = self._load_data(y_true=y_true, group_a=group_a, group_b=group_b)
        y_true = params["y_true"]
        group_a = params["group_a"]
        group_b = params["group_b"]
        sensitive_features = np.stack([group_a, group_b], axis=1)
        self.classes_ = params["classes_"]

        self.algorithm.fit(
            X=X, y=y_true, sensitive_features=sensitive_features, random_state=self.seed
        )
        return self

    def predict(self, X: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """
        return self.algorithm.predict(X)

    def predict_proba(self, X: np.ndarray):
        """
        Probability Prediction

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        Returns
        -------
        numpy.ndarray
            probability output per sample.
        """
        return self.algorithm.predict_proba(X)
