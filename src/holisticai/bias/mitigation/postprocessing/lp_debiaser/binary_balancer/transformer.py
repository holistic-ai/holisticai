from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from holisticai.bias.mitigation.postprocessing.lp_debiaser.binary_balancer.algorithm import BinaryBalancerAlgorithm
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from holisticai.utils.transformers.bias import SensitiveGroups


class LPDebiaserBinary(BMPost):
    """
    Linear Programmin Debiaser is a postprocessing algorithms designed to debias pretrained classifiers.\
    The algorithm use constraints such as Equalized Odds and Equalized Opportunity.

    Parameters
    ----------
    constraint : str
        Strategy used to evalute the cost function  The available contraints  are: "EqualizedOdds", "EqualizedOpportunity"

    References
    ----------
        .. [1] Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning."\
        Advances in neural information processing systems 29 (2016).
    """

    CONSTRAINT = Literal["EqualizedOdds", "EqualizedOpportunity"]
    OBJ_FUN = Literal["macro", "micro"]

    def __init__(
        self,
        constraint: Optional[CONSTRAINT] = "EqualizedOdds",
    ):
        self.constraint = constraint
        self._sensgroups = SensitiveGroups()

    def fit(
        self,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ):
        """
        Compute parameters for Linear Programming Debiaser.\
        For binary classification y_pred or y_proba can be used.\
        For Multiclass classification only y_pred must be used.

        Parameters
        ----------
        y : array-like
            Target vector
        y_pred : array-like
            Predicted label vector (num_examples,).
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability\
            estimates must sum to 1 across the possible classes and each matrix value\
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        Self
        """
        params = self._load_data(
            y=y,
            y_pred=y_pred,
            y_proba=y_proba,
            group_a=group_a,
            group_b=group_b,
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y = params["y"]
        y_pred = params.get("y_pred", None)
        y_proba = params.get("y_proba", None)

        sensitive_features = np.stack([group_a, group_b], axis=1)
        p_attr = self._sensgroups.fit_transform(sensitive_features, convert_numeric=True)

        constraints_catalog, objective_catalog = self._get_catalogs()

        constraint = constraints_catalog[self.constraint]()

        objective = objective_catalog["losses"]()

        self.algorithm = BinaryBalancerAlgorithm(constraint=constraint, objective=objective)

        self.algorithm.fit(y_true=y, y_pred=y_pred, y_proba=y_proba, p_attr=p_attr)

        return self

    def transform(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ):
        """
        Apply transform function to predictions and likelihoods

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability\
            estimates must sum to 1 across the possible classes and each matrix value\
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        dict
            A dictionary of new predictions
        """

        params = self._load_data(y_pred=y_pred, y_proba=y_proba, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1

        sensitive_features = np.stack([group_a, group_b], axis=1)
        p_attr = self._sensgroups.transform(sensitive_features, convert_numeric=True)

        y_proba = params.get("y_proba", None)
        y_pred = params.get("y_pred", None)
        new_y_pred = self.algorithm.predict(y_pred=y_pred, y_proba=y_proba, p_attr=p_attr)
        return {"y_pred": new_y_pred}

    def fit_transform(
        self,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ):
        """
        Fit and transform

        Parameters
        ----------
        y : array-like
            Target vector
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability\
            estimates must sum to 1 across the possible classes and each matrix value\
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        dict
            A dictionary of new predictions
        """
        return self.fit(
            y=y,
            y_proba=y_proba,
            y_pred=y_pred,
            group_a=group_a,
            group_b=group_b,
        ).transform(y_proba=y_proba, y_pred=y_pred, group_a=group_a, group_b=group_b)

    def _get_catalogs(self):
        from holisticai.bias.mitigation.postprocessing.lp_debiaser.binary_balancer.constraints import (
            EqualizedOdds,
            EqualizedOpportunity,
            Losses,
        )

        cons_cat = {
            "EqualizedOdds": EqualizedOdds,
            "EqualizedOpportunity": EqualizedOpportunity,
        }

        obj_cat = {
            "losses": Losses,
        }

        return cons_cat, obj_cat
