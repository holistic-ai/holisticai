import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from ..commons._conventions import *
from ._lagrangian import Lagrangian


class ExponentiatedGradientAlgorithm:
    """
    This class implements the exponentiated gradient approach to reductions.

    Reference
    ---------
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`.

    """

    def __init__(
        self,
        estimator,
        constraints,
        eps: Optional[float] = 0.01,
        max_iter: Optional[int] = 50,
        nu: Optional[float] = None,
        eta0: Optional[float] = 2.0,
        verbose: Optional[int] = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Exponentiated Gradient Reduction Algorithm

        Parameters
        ----------
        estimator : estimator
            An estimator implementing methods :code:`fit(X, y, sample_weight)` and
            :code:`predict(X)`, where `X` is the matrix of features, `y` is the
            vector of labels (binary classification) or continuous values
            (regression), and `sample_weight` is a vector of weights.
            In binary classification labels `y` and predictions returned by
            :code:`predict(X)` are either 0 or 1.
            In regression values `y` and predictions are continuous.

        constraints : ClasificationConstraint
            The disparity constraints expressed as moments

        eps : float
            Allowed fairness constraint violation; the solution is guaranteed to
            have the error within :code:`2*best_gap` of the best error under
            constraint `eps`; the constraint violation is at most
            :code:`2*(eps+best_gap)`

        max_iter : int
            Maximum number of iterations

        nu : float
            Convergence threshold for the duality gap, corresponding to a
            conservative automatic setting based on the statistical uncertainty
            in measuring classification error

        eta_0 : float
            Initial setting of the learning rate

        verbose : int
            If >0, will show progress percentage.
        """
        self.estimator = estimator
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.seed = seed
        self.monitor = Monitor(verbose=verbose)
        self.eg_helper = Helper()

    def fit(self, X, y, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y : array-like
            Target vector
        """
        B = 1 / self.eps

        lagrangian = Lagrangian(X, y, self.estimator, self.constraints, B, **kwargs)

        theta = pd.Series(0, lagrangian.constraints.index)

        def compute_default_nu(h):
            absolute_error = (h(X) - self.constraints._y_as_series).abs()
            nu = (
                ACCURACY_MUL
                * absolute_error.std()
                / np.sqrt(self.constraints.total_samples)
            )
            return nu

        eta = self.eta0 / B
        gap_LP = np.PINF
        Q_LP = None
        nu = self.nu
        last_regret_checked = REGRET_CHECK_START_T
        last_gap = np.PINF
        self.monitor.max_iter = self.max_iter
        for t in range(0, self.max_iter):

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            h, h_idx = lagrangian.best_h(lambda_vec)
            nu = compute_default_nu(h) if (t == 0 and nu is None) else nu
            Q_EG, result_EG = self.eg_helper.compute_EG(
                t, h_idx, lambda_vec, lagrangian, nu
            )
            gap_EG = result_EG.gap()
            gamma = lagrangian.gammas[h_idx]

            if t > 0:
                Q_LP, result_LP = self.eg_helper.compute_LP(t, lagrangian, nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            self.monitor.update(gap_EG, Q_EG, gap_LP, Q_LP)

            if self.monitor.stop_condition(t, nu):
                break

            # update regret
            if t >= last_regret_checked * REGRET_CHECK_INCREASE_T:
                best_gap = min(self.monitor.gaps_EG)

                if best_gap > last_gap * SHRINK_REGRET:
                    eta *= SHRINK_ETA

                last_regret_checked = t
                last_gap = best_gap

            # update theta based on learning rate
            theta += eta * (gamma - self.constraints.bound())

            self.monitor.log_progress()

        # retain relevant result data
        report = self.monitor.make_a_summary(lagrangian)
        for param in ["_hs", "weights_"]:
            setattr(self, param, report[param])
        return self

    def predict(self, X):
        """
        Provide predictions for the given input data.

        Description
        -----------
        Predictions are randomized, i.e., repeatedly calling `predict` with
        the same feature data may yield different output. This
        non-deterministic behavior is intended and stems from the nature of
        the exponentiated gradient algorithm.

        Parameters
        ----------
        X : matrix-like
            Feature data

        Returns
        -------
        Scalar or vector
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        random_state = check_random_state(self.seed)

        if self.constraints.PROBLEM_TYPE == "classification":
            positive_probs = self.predict_proba(X)[:, 1]
            return (positive_probs >= random_state.rand(len(positive_probs))) * 1
        else:
            pred = self._forward(X)
            randomized_pred = np.zeros(pred.shape[0])
            for i in range(pred.shape[0]):
                randomized_pred[i] = random_state.choice(
                    pred.iloc[i, :], p=self.weights_
                )
            return randomized_pred

    def predict_proba(self, X):
        """
        Probability mass function for the given input data.

        Description
        -----------
        For each data point, provide the probabilities with which 0 and 1 is
        returned as a prediction.

        Parameters
        ----------
        X : matrix-like
            Feature data

        Returns
        -------
        pandas.DataFrame
            Array of tuples with the probabilities of predicting 0 and 1.
        """
        pred = self._forward(X)
        positive_probs = pred[self.weights_.index].dot(self.weights_).to_frame()
        return np.concatenate((1 - positive_probs, positive_probs), axis=1)

    def _forward(self, X):
        pred = pd.DataFrame()
        for t in range(len(self._hs)):
            if self.weights_[t] == 0:
                pred[t] = np.zeros(len(X))
            else:
                pred[t] = self._hs[t](X)
        return pred


class Helper:
    """
    A helper class to handle historical data during each iteartion.
    """

    def __init__(self):
        self.Qsum = pd.Series(dtype="float64")
        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()

    def update_Qsum(self, h_idx):
        if h_idx not in self.Qsum.index:
            self.Qsum.at[h_idx] = 0.0
        self.Qsum[h_idx] += 1.0

    def compute_EG(self, t, h_idx, lambda_vec, lagrangian, nu):
        self.lambda_vecs_EG_[t] = lambda_vec
        lambda_EG = self.lambda_vecs_EG_.mean(axis=1)
        self.update_Qsum(h_idx)
        Q_EG = self.Qsum / self.Qsum.sum()
        result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, nu)
        return Q_EG, result_EG

    def compute_LP(self, t, lagrangian, nu):
        Q_LP, lambda_LP, result_LP = lagrangian.solve_linprog(nu)
        self.lambda_vecs_LP_[t] = lambda_LP
        return Q_LP, result_LP


class Monitor:
    """Monitor class used to store and create a summary report"""

    def __init__(self, verbose):
        self.gaps = []
        self.Qs = []
        self.gaps_EG = []
        self.verbose = verbose
        self.step = 0

    def update(self, gap_EG, Q_EG, gap_LP, Q_LP):
        self.gaps_EG.append(gap_EG)

        if gap_EG < gap_LP:
            self.Qs.append(Q_EG)
            self.gaps.append(gap_EG)
        else:
            self.Qs.append(Q_LP)
            self.gaps.append(gap_LP)

    def stop_condition(self, t, nu):
        return (self.gaps[t] < nu) and (t >= MIN_ITER)

    def make_a_summary(self, lagrangian):
        report = {}
        gaps_series = pd.Series(self.gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + PRECISION]

        report["best_iter_"] = best_iter_ = gaps_best.index[-1]
        report["best_gap_"] = self.gaps[best_iter_]
        report["last_iter_"] = len(self.Qs) - 1
        report["weights_"] = weights_ = self.Qs[best_iter_]
        report["_hs"] = _hs = lagrangian.hs

        for h_idx in _hs.index:
            if h_idx not in weights_.index:
                report["weights_"].at[h_idx] = 0.0

        report["predictors_"] = lagrangian.predictors
        report["lambda_vecs_"] = lagrangian.lambdas

        return report

    def log_progress(self):
        self.step += 1
        if self.verbose:
            sys.stdout.write(f"\rsteps: {self.step}\tBest gap:{min(self.gaps_EG):.4f}")
            sys.stdout.flush()
