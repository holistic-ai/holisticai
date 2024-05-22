from calendar import c
from typing import Optional

import numpy as np

from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithm import MulticlassBalancerAlgorithm


class LPDebiaserMulticlass(BMPost):
    """
    Linear Programmin Debiaser is a postprocessing algorithms designed to debias pretrained classifiers.
    The algorithm use constraints such as Equalized Odds and Equalized Opportunity.
    This technique extends LPDebiaserBinary for multiclass classification.
    References:
        Putzel, Preston, and Scott Lee. "Blackbox Post-Processing for Multiclass Fairness."
        arXiv preprint arXiv:2201.04461 (2022).
    """

    CONSTRAINT = ["EqualizedOdds", "EqualizedOpportunity"]
    OBJ_FUN = ["macro", "micro"]

    def __init__(
        self,
        constraint: Optional["CONSTRAINT"] = "EqualizedOdds",
        loss: Optional["OBJ_FUN"] = "macro",
    ):
        """
        Parameters
        ----------
        constraint : str
            Strategy used to evalute the cost function  The available contraints  are:
            [
                "EqualizedOdds",
                "EqualizedOpportunity"
            ]

        loss : str
            The loss function to optimize:
            [
                "macro",
                "micro"
            ],
            default "macro"
        """
        self.constraint = constraint
        self.loss = loss
        self.sens_groups = SensitiveGroups()

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """

        Description
        ----------
        Compute parameters for Linear Programming Debiaser.
        For binary classification y_pred or y_proba can be used.
        For Multiclass classification only y_pred must be used.
        Parameters
        ----------
        y_true : array-like
            Target vector
        y_pred : array-like
            Predicted label vector (num_examples,).
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
        Self
        """
        params = self._load_data(
            y_true=y_true, y_pred=y_pred, group_a=group_a, group_b=group_b
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_true = params["y_true"]
        y_pred = params["y_pred"]

        sensitive_features = np.stack([group_a, group_b], axis=1)
        p_attr = self.sens_groups.fit_transform(
            sensitive_features, convert_numeric=True
        )

        constraints_catalog, objective_catalog = self._get_catalogs()

        constraint = constraints_catalog[self.constraint]()

        objective = objective_catalog[self.loss]()

        self.algorithm = MulticlassBalancerAlgorithm(
            constraint=constraint, objective=objective
        )

        self.algorithm.fit(y_true=y_true, y_pred=y_pred, p_attr=p_attr)
        return self

    def transform(
        self,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Apply transform function to predictions and likelihoods

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        dictionnary with new predictions
        """

        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]

        sensitive_features = np.stack([group_a, group_b], axis=1)
        p_attr = self.sens_groups.transform(sensitive_features, convert_numeric=True)
        new_y_pred = self.algorithm.predict(y_pred=y_pred, p_attr=p_attr)
        return {"y_pred": new_y_pred}

    def fit_transform(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit and transform

        Parameters
        ----------
        y_true : array-like
            Target vector
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        dictionnary with new predictions
        """
        return self.fit(
            y_true=y_true,
            y_pred=y_pred,
            group_a=group_a,
            group_b=group_b,
        ).transform(y_pred=y_pred, group_a=group_a, group_b=group_b)

    def _get_catalogs(self):
        from .constraints import (
            EqualizedOdds,
            EqualizedOpportunity,
            MacroLosses,
            MicroLosses,
            Strict,
        )

        cons_cat = {
            "EqualizedOdds": EqualizedOdds,
            "EqualizedOpportunity": EqualizedOpportunity,
            "Strict": Strict,
        }

        obj_cat = {"macro": MacroLosses, "micro": MicroLosses}

        return cons_cat, obj_cat
