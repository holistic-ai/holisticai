from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from ..commons.classification import _constraints as cc
from ..commons.regression import _constraints as rc
from ..commons.regression import _losses as rl
from ._grid_generator import GridGenerator
from .algorithm import GridSearchAlgorithm


class GridSearchReduction(BaseEstimator, BMImp):
    """
    Grid search technique can be used for fair classification or fair regression.
    - For classification it reduces fair classification to a sequence of cost-sensitive classification problems,
    returning the deterministic classifier with the lowest empirical error subject to fair classification constraints among the
    candidates searched.
    - For regression it uses the same priniciple to return a deterministic regressor with the lowest empirical error subject to the
    constraint of bounded group loss.

    References:
        Agarwal, Alekh, et al. "A reductions approach to fair classification."
        International Conference on Machine Learning. PMLR, 2018.

        Agarwal, Alekh, Miroslav Dudík, and Zhiwei Steven Wu.
        "Fair regression: Quantitative definitions and reduction-based algorithms."
        International Conference on Machine Learning. PMLR, 2019.
    """

    CONSTRAINTS = [
        "DemographicParity",
        "EqualizedOdds",
        "TruePositiveRateParity",
        "FalsePositiveRateParity",
        "ErrorRateParity",
        "BoundedGroupLoss",
    ]

    def __init__(
        self,
        constraints: str = "EqualizedOdds",
        constraint_weight: Optional[float] = 0.5,
        loss: str = "ZeroOne",
        min_val: float = None,
        max_val: float = None,
        grid_size: Optional[int] = 10,
        grid_limit: Optional[float] = 2.0,
        verbose: Optional[int] = 0.0,
    ):
        """
        Init Grid Search Reduction Transformer

        Parameters
        ----------

        constraints : string
            The disparity constraints expressed as string:
                - "DemographicParity",
                - "EqualizedOdds",
                - "TruePositiveRateParity",
                - "FalsePositiveRateParity",
                - "ErrorRateParity"
                - "BoundedGroupLoss"

        constraint_weight : float
            Specifies the relative weight put on the constraint violation when selecting the
            best model. The weight placed on the error rate will be :code:`1-constraint_weight`

        loss : str
            String identifying loss function for constraints. Options include "ZeroOne", "Square", and "Absolute."

        min_val : float
            Loss function parameter for "Square" and "Absolute," typically the minimum of the range of y values.

        max_val: float
            Loss function parameter for "Square" and "Absolute," typically the maximum of the range of y values.

        grid_size : int
            The number of Lagrange multipliers to generate in the grid

        grid_limit : float
            The largest Lagrange multiplier to generate. The grid will contain
            values distributed between :code:`-grid_limit` and :code:`grid_limit`
            by default

        verbose : int
            If >0, will show progress percentage.

        """
        self.constraints = constraints
        self.constraint_weight = constraint_weight
        self.loss = loss
        self.min_val = min_val
        self.max_val = max_val
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.verbose = verbose

    def transform_estimator(self, estimator):
        self.estimator = estimator
        return self

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """

        Fit model using Grid Search Reduction.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
            Self
        """

        params = self._load_data(X=X, y_true=y_true, group_a=group_a, group_b=group_b)
        group_a = params["group_a"]
        group_b = params["group_b"]
        X = params["X"]
        y_true = params["y_true"]

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

        self.model_ = GridSearchAlgorithm(
            estimator=self.estimator_,
            constraint=self.constraint_,
            constraint_weight=self.constraint_weight,
            grid_size=self.grid_size,
            grid_limit=self.grid_limit,
            verbose=self.verbose,
        )

        self.model_.fit(X, y_true, sensitive_features=sensitive_features)

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
            kargs.update({"loss": self.loss_})
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
            Input Matrix

        Returns
        -------
        numpy.ndarray: Predicted output
        """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Probability Prediction

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : matrix-like
            Input Matrix

        Returns
        -------
        numpy.ndarray
            probability output
        """
        return self.model_.predict_proba(X)
