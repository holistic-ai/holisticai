from __future__ import annotations

from typing import Literal

import numpy as np
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from scipy.optimize import linprog
from sklearn.metrics import confusion_matrix


class EqualizedOdds(BMPost):
    """
    Equalized odds postprocessing use linear programming to find the probability with which change\
    favorable labels (y=1) to unfavorable labels (y=0) in the output estimator to optimize equalized odds.

    Parameters
    ----------
    solver : str
        Algorithm name used to solve the standard form problem. Solver supported must depend of your scipy poackage version.
        for scipy 1.9.0 the solvers available are:
        ["highs", "highs-ds", "highs-ipm", "interior-point", "revised simplex", "simplex"]

    seed : int
        Random seed for repeatability.

    References
    ----------
        .. [1] Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning."\
        Advances in neural information processing systems 29 (2016).

        .. [2] Pleiss, Geoff, et al. "On fairness and calibration."\
        Advances in neural information processing systems 30 (2017).
    """

    SOLVERS = Literal[
        "highs",
        "highs-ds",
        "highs-ipm",
        "interior-point",
        "revised simplex",
        "simplex",
    ]

    def __init__(self, solver: str | None = "highs", seed: int | None = 42):
        self.solver = solver
        self.seed = seed

    def _objective_function_parameters_by_group(self, labels, predictions, group):
        # Compute Equalized Odds algorithm parameters for specific group

        labels = labels[group]
        predictions = predictions[group]

        prev = predictions
        flip_prev = 1 - predictions

        flip_predictions = 1 - predictions
        flip_labels = 1 - labels
        base_rate = np.mean(labels)

        tnr, fpr, fnr, tpr = confusion_matrix(labels, predictions, normalize="true").ravel()

        m_tn = np.logical_and(flip_predictions, flip_labels)
        m_fn = np.logical_and(flip_predictions, labels)
        m_fp = np.logical_and(predictions, flip_labels)
        m_tp = np.logical_and(predictions, labels)

        c = np.array([fpr - tpr, tnr - fnr])

        true_positive = (np.mean(prev * m_tp) - np.mean(flip_prev * m_tp)) / base_rate
        false_negative = (np.mean(flip_prev * m_fn) - np.mean(prev * m_fn)) / base_rate
        false_positive = (np.mean(prev * m_fp) - np.mean(flip_prev * m_fp)) / (1 - base_rate)
        true_negative = (np.mean(flip_prev * m_tn) - np.mean(prev * m_tn)) / (1 - base_rate)

        A_eq = np.array([[true_positive, false_negative], [false_positive, true_negative]])

        b_tp_fn = (np.mean(flip_prev * m_tp) + np.mean(prev * m_fn)) / base_rate
        b_fp_tn = (np.mean(flip_prev * m_fp) + np.mean(prev * m_tn)) / (1 - base_rate)

        b_eq = np.array([b_tp_fn, b_fp_tn])

        return {"c": c, "A_eq": A_eq, "b_eq": b_eq}

    def _build_objective_function(self, params_group_a, params_group_b):
        """
        Concat parameters from group a and b to create the objective function

        Parameters
        ----------
        params_group_a : dict
            dictionary of parameters with A_eq, b_eq, C keys.
        params_group_b : dict
            dictionary of parameters with A_eq, b_eq, C keys.

        Returns
        -------
        Tuple of objetive function parameters (A_eq, b_eq, C)
        """
        A_eq = np.concatenate([params_group_a["A_eq"], -params_group_b["A_eq"]], axis=1)
        b_eq = params_group_b["b_eq"] - params_group_a["b_eq"]
        C = np.concatenate([params_group_a["c"], params_group_b["c"]], axis=0)
        return A_eq, b_eq, C

    def _adjust_fairness(self, predictions, likelihoods, n2p, p2p):
        """
        Invert predictions and likelihoods with some probabilities

        Description
        ----------
        Use n2p and (1 - p2p) to invert output labels

        Parameters
        ----------
        predictions : numpy array
            binary predicted vector
        likelihoods : numpy array
            predicted score vector
        n2p : numpy array
            probability vector
        p2p : numpy array
            probability vector

        Returns
        -------
        numpy array
            new prediction vector with some inverted values

        numpy array
            new score vector with some inverted values
        """

        fair_predictions = predictions.copy()
        pos_indices = np.where(predictions == 1)[0]
        neg_indices = np.where(predictions == 0)[0]

        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        n2p_indices = neg_indices[: int(len(neg_indices) * n2p)]
        fair_predictions[n2p_indices] = 1

        p2n_indices = pos_indices[: int(len(pos_indices) * (1 - p2p))]
        fair_predictions[p2n_indices] = 0

        fair_likelihoods = likelihoods.copy()
        fair_likelihoods[n2p_indices] = 1 - fair_likelihoods[n2p_indices]
        fair_likelihoods[p2n_indices] = 1 - fair_likelihoods[p2n_indices]

        return fair_predictions, fair_likelihoods

    def fit(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> EqualizedOdds:
        """
        Compute parameters for equalizing odds.

        Description
        ----------
        Build parameters for the objetive function and call the solver to find the algorithm parameters.

        Parameters
        ----------
        y : array-like
            Target vector (nb_examlpes,)
        y_pred : array-like
            Predicted vector (nb_examlpes,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
            Self
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        params = self._load_data(y=y, y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y = params["y"]
        y_pred = params["y_pred"]

        parameters_group_a = self._objective_function_parameters_by_group(y, y_pred, group_a)
        parameters_group_b = self._objective_function_parameters_by_group(y, y_pred, group_b)
        A_eq, b_eq, C = self._build_objective_function(parameters_group_a, parameters_group_b)

        A_ub = np.array(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ],
            dtype=np.float64,
        )

        b_ub = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)

        # Linear program
        self.model_params = linprog(C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method=self.solver)

        return self

    def transform(self, y_pred: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Apply transform function to predictions and likelihoods

        Description
        ----------
        Use the fitted probability to change the output label and invert the likelihood

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
        dict
            A dictionary with two keys, y_pred and y_score, which refers to the predicted labels and their probabilities, respectively.
        """
        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]
        likelihoods = y_pred.copy().astype(np.float64)

        sp2p, sn2p, op2p, on2p = self.model_params.x

        predictions_a = y_pred[group_a]
        predictions_b = y_pred[group_b]
        likelihoods_a = likelihoods[group_a]
        likelihoods_b = likelihoods[group_b]

        # Randomly flip labels according to the probabilities in model_params
        fair_predictions_a, fair_likelihoods_a = self._adjust_fairness(predictions_a, likelihoods_a, sn2p, sp2p)
        fair_predictions_b, fair_likelihoods_b = self._adjust_fairness(predictions_b, likelihoods_b, on2p, op2p)

        # Mutated, fairer dataset with new labels4
        new_y_pred = y_pred.copy()
        new_y_pred[group_a] = fair_predictions_a
        new_y_pred[group_b] = fair_predictions_b

        new_y_score = likelihoods.copy()
        new_y_score[group_a] = fair_likelihoods_a
        new_y_score[group_b] = fair_likelihoods_b

        return {"y_pred": new_y_pred, "y_score": new_y_score}
