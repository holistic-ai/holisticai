from typing import Optional

import numpy as np
from sklearn.metrics._classification import _check_zero_division, _prf_divide

from holisticai.utils.transformers.bias import BMPostprocessing as BMPost


class CalibratedEqualizedOdds(BMPost):
    """
    Calibrated equalized odds postprocessing optimizes over calibrated classifier score outputs to find
    probabilities with which to change output labels with an equalized odds objective.
    References:
        Pleiss, Geoff, et al. "On fairness and calibration."
        Advances in neural information processing systems 30 (2017).
    """

    COST_CONSTRAINT = ["fnr", "fpr", "weighted"]

    def __init__(
        self,
        cost_constraint: Optional[str] = "fnr",
        alpha: Optional[float] = None,
    ):
        """
        Create a Calibrated Equalized Odds Post-processing instance.

        Parameters
        ----------
        cost_constraint : str
            Strategy used to evalute the cost function  The available contraints  are:
            [
                "fnr",
                "fpr",
                "weighted"
            ]
            false negative rate (fnr), false positive rate (fpr), and weighted

        alpha : float
            Used only with cost contraint  "weighted".
            Value between 0 and 1 used to combine fnr and fpr cost constraint.
        """
        self.cost_constraint = cost_constraint
        self.alpha = alpha

    def _build_cost(self, y_true, y_score, base_rate, sample_weight):
        if self.cost_constraint == "fpr":
            alpha = 0
        elif self.cost_constraint == "fnr":
            alpha = 1
        elif self.cost_constraint == "weighted":
            alpha = base_rate if self.alpha is None else self.alpha
        else:
            raise ValueError(f"unknown cost constraint: {self.cost_constraint}")

        gfpr = generalized_fpr(y_true, y_score, sample_weight=sample_weight)
        gfnr = generalized_fnr(y_true, y_score, sample_weight=sample_weight)

        return (1 - alpha) * gfpr + alpha * gfnr

    def _build_cost_variables(self, y_true, y_score, sample_weight):
        base_rate = y_true.mean()
        build_cost = lambda score: self._build_cost(
            y_true=y_true,
            y_score=score,
            base_rate=base_rate,
            sample_weight=sample_weight,
        )

        cost = build_cost(y_score)
        trivial_cost = build_cost(np.full_like(y_score, base_rate))
        return base_rate, cost, trivial_cost

    def _mitigate_bias_score(self, y_score, group, mix_rate, base_rate):
        indexes = np.random.random(sum(group)) <= mix_rate
        new_y_score = y_score[group == 1].copy()
        new_y_score[indexes] = base_rate
        return new_y_score

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Compute parameters for calibrated equalized odds.

        Description
        ----------
        Compute parameters for calibrated equalized odds algorithm.

        Parameters
        ----------
        y_true : array-like
            Target vector
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability
            estimates must sum to 1 across the possible classes and each matrix value
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight : array-like
            Sample weights. Used to weight generalized false positive rate (GFPR) and
            generalized false negative rate (GFNR). Sample weights could be used during training
            or computed by previous preprocessing strategy.
        Returns
        -------
        Self
        """
        params = self._load_data(
            y_true=y_true,
            y_proba=y_proba,
            group_a=group_a,
            group_b=group_b,
            sample_weight=sample_weight,
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_true = params["y_true"]
        y_score = params["y_score"]
        sample_weight = params["sample_weight"]

        self.base_rate_a, a_cost, a_trivial_cost = self._build_cost_variables(
            y_true[group_a], y_score[group_a], sample_weight=sample_weight[group_a]
        )

        self.base_rate_b, b_cost, b_trivial_cost = self._build_cost_variables(
            y_true[group_b], y_score[group_b], sample_weight=sample_weight[group_b]
        )

        b_costs_more = b_cost > a_cost
        self.a_mix_rate = (
            (b_cost - a_cost) / (a_trivial_cost - a_cost) if b_costs_more else 0
        )
        self.b_mix_rate = (
            0 if b_costs_more else (a_cost - b_cost) / (b_trivial_cost - b_cost)
        )

        return self

    def transform(
        self,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        threshold: Optional[float] = 0.5,
    ):
        """
        Apply transform function to predictions and likelihoods

        Description
        ----------
        Use a fitted probability to change the output label and invert the likelihood

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability
            estimates must sum to 1 across the possible classes and each matrix value
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        threshold : float
            float value to discriminate between 0 and 1

        Returns
        -------
        dictionnary with new predictions
        """

        params = self._load_data(
            y_pred=y_pred, y_proba=y_proba, group_a=group_a, group_b=group_b
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]
        y_score = params["y_score"]

        new_y_score_a = self._mitigate_bias_score(
            y_score, group_a, self.a_mix_rate, self.base_rate_a
        )
        new_y_score_b = self._mitigate_bias_score(
            y_score, group_b, self.b_mix_rate, self.base_rate_b
        )

        new_y_score = y_score.copy()
        new_y_score[group_a] = new_y_score_a
        new_y_score[group_b] = new_y_score_b

        new_y_pred = y_pred.copy()
        new_y_pred[group_a] = np.where(new_y_score[group_a] >= threshold, 1, 0)
        new_y_pred[group_b] = np.where(new_y_score[group_b] >= threshold, 1, 0)

        return {
            "y_pred": new_y_pred,
            "y_score": new_y_score,
        }

    def fit_transform(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        threshold: Optional[float] = 0.5,
    ):
        """
        Fit and transform

        Description
        ----------
        Fit and transform

        Parameters
        ----------
        y_true : array-like
            Target vector
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability
            estimates must sum to 1 across the possible classes and each matrix value
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight : array-like
            Sample weights. Used to weight generalized false positive rate (GFPR) and
            generalized false negative rate (GFNR). Sample weights could be used during training
            or computed by previous preprocessing strategy.

        threshold : float
            float value to discriminate between 0 and 1

        Returns
        -------
        dictionnary with new predictions
        """
        return self.fit(
            y_true,
            y_proba,
            group_a,
            group_b,
            sample_weight,
        ).transform(y_true, y_proba, group_a, group_b, threshold)


def generalized_fpr(y_true, y_score, sample_weight):
    neg_idx = y_true != 1
    neg_weights = sample_weight[neg_idx]
    neg = neg_weights.sum()
    fpr = (y_score[neg_idx] * neg_weights).sum() / neg
    return fpr


def generalized_fnr(y_true, y_score, sample_weight):
    pos_idx = y_true == 1
    pos_weights = sample_weight[pos_idx]
    pos = pos_weights.sum()
    fnr = ((1 - y_score)[pos_idx] * pos_weights).sum() / pos
    return fnr
