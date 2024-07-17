from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from holisticai.bias.mitigation.inprocessing.commons.classification import _constraints as cc
from holisticai.bias.mitigation.inprocessing.commons.regression import _constraints as rc
from holisticai.bias.mitigation.inprocessing.commons.regression import _losses as rl
from holisticai.bias.mitigation.inprocessing.exponentiated_gradient.algorithm import ExponentiatedGradientAlgorithm
from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ExponentiatedGradientReduction(BaseEstimator, ClassifierMixin, BMImp):
    """Exponentiated gradient reduction

    Exponentiated gradient reduction is an in-processing technique that reduces\
    fair classification to a sequence of cost-sensitive classification problems,\
    returning a randomized classifier with the lowest empirical error subject to\
    fair classification constraints.

    Parameters
    ----------

        estimator : sklearn-like
            The model you want to mitigate bias for.

        constraints: str
            The disparity constraints:
                - "DemographicParity"
                - "EqualizedOdds"
                - "TruePositiveRateParity"
                - "FalsePositiveRateParity"
                - "ErrorRateParity"

        eps: float
            Allowed fairness constraint violation; the solution is\
            guaranteed to have the error within ``2*best_gap`` of the best\
            error under constraint eps; the constraint violation is at most\
            ``2*(eps+best_gap)``.

        num_iter: int
            Maximum number of iterations.

        nu: float
            Convergence threshold for the duality gap, corresponding to a\
            conservative automatic setting based on the statistical\
            uncertainty in measuring classification error.

        eta_mul: float
            Initial setting of the learning rate.

        drop_prot_attr: bool
            Boolean flag indicating whether to drop protected\
            attributes from training data.

        loss : str
            String identifying loss function for constraints. Options include "ZeroOne", "Square", and "Absolute."

        min_val : float
            Loss function parameter for "Square" and "Absolute," typically the minimum of the range of y values.

        max_val: float
            Loss function parameter for "Square" and "Absolute," typically the maximum of the range of y values.

        verbose : int
            If >0, will show progress percentage.

        seed: int
            seed for random initialization

    References
    ---------
    .. [1] Agarwal, Alekh, et al. "A reductions approach to fair classification."
        International Conference on Machine Learning. PMLR, 2018.
    """

    CONSTRAINTS = Literal[
        "DemographicParity",
        "EqualizedOdds",
        "TruePositiveRateParity",
        "FalsePositiveRateParity",
        "ErrorRateParity",
    ]

    def __init__(
        self,
        constraints: str = "EqualizedOdds",
        eps: Optional[float] = 0.01,
        max_iter: Optional[int] = 50,
        nu: Optional[float] = None,
        eta0: Optional[float] = 2.0,
        loss: str = "ZeroOne",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        upper_bound: float = 0.01,
        verbose: Optional[int] = 0,
        estimator=None,
        seed: int = 0,
    ):
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.loss = loss
        self.min_val = min_val
        self.max_val = max_val
        self.upper_bound = upper_bound
        self.verbose = verbose
        self.estimator = estimator
        self.seed = seed

    def transform_estimator(self, estimator):
        """
        This method is deprecated but retained for backwards-compatibility. You should pass the estimator object directly in the constructor.
        """

        self.estimator = estimator
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit Exponentiated Gradient Reduction

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        Self

        """
        params = self._load_data(X=X, y=y, group_a=group_a, group_b=group_b)
        group_a = params["group_a"]
        group_b = params["group_b"]
        X = params["X"]
        y = params["y"]

        sensitive_features = np.stack([group_a, group_b], axis=1)

        self.estimator_ = clone(self.estimator)

        constraints_catalog = {
            "DemographicParity": cc.DemographicParity,
            "EqualizedOdds": cc.EqualizedOdds,
            "TruePositiveRateParity": cc.TruePositiveRateParity,
            "FalsePositiveRateParity": cc.FalsePositiveRateParity,
            "ErrorRateParity": cc.ErrorRateParity,
            "BoundedGroupLoss": rc.BoundedGroupLoss,
        }

        constraint_kargs = self._constraint_parameters()
        self.constraint_ = constraints_catalog[self.constraints](**constraint_kargs)

        self.model_ = ExponentiatedGradientAlgorithm(
            self.estimator_,
            constraints=self.constraint_,
            eps=self.eps,
            max_iter=self.max_iter,
            nu=self.nu,
            eta0=self.eta0,
            verbose=self.verbose,
            seed=self.seed,
        )

        self.model_.fit(X, y, sensitive_features=sensitive_features)

        return self

    def _constraint_parameters(self):
        kargs = {}
        if self.constraints == "BoundedGroupLoss":
            losses = {
                "ZeroOne": rl.ZeroOneLoss,
                "Square": rl.SquareLoss,
                "Absolute": rl.AbsoluteLoss,
            }
            if self.loss == "ZeroOne":
                self.loss_ = losses[self.loss]()
            else:
                self.loss_ = losses[self.loss](self.min_val, self.max_val)
            kargs.update({"loss": self.loss_, "upper_bound": self.upper_bound})
        return kargs

    def predict(self, X):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input matrix.

        Returns
        -------
        numpy.ndarray: Predicted output
        """
        return self.model_.predict(X).ravel()

    def predict_proba(self, X):
        """
        Predict Probabilities

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input matrix.

        Returns
        -------
        numpy.ndarray: probability output
        """
        return self.model_.predict_proba(X)
