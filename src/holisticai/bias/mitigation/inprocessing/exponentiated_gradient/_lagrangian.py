import numpy as np
import pandas as pd
import scipy.optimize as opt
from holisticai.bias.mitigation.inprocessing.commons._conventions import PRECISION
from sklearn import clone


class Lagrangian:
    """Operations related to the Lagrangian"""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, estimator, constraints, B: float, opt_lambda: bool = True, **kwargs
    ):
        """
        Initialize Lagrangian object.

        Parameters
        ----------
        X :  matrix-like
            the training features
        sensitive_features :  array-like
            the sensitive features to use for constraints
        y :  array-like
            the training labels
        estimator :
            the estimator to fit in every iteration of :meth:`best_h` using a
            :meth:`fit` method with arguments `X`, `y`, and `sample_weight`
        constraints : ClassificationConstraint
            Object describing the parity constraints. This provides the reweighting
            and relabelling.
        B : float
            bound on the L1-norm of the lambda vector
        opt_lambda : bool
            indicates whether to optimize lambda during the calculation of the
            Lagrangian; optional with default value True
        """
        self.constraints = constraints
        self.constraints.load_data(X, y, **kwargs)
        self.obj = self.constraints.default_objective()
        self.obj.load_data(X, y, **kwargs)
        self.estimator = estimator
        self.B = B
        self.opt_lambda = opt_lambda
        self.hs = pd.Series(dtype="float64")
        self.predictors = pd.Series(dtype="float64")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None

    def _eval(self, Q, lambda_vec):
        """
        Return the value of the Lagrangian.

        Parameters
        ----------
        Q : {pandas.Series, callable}
            `Q` is either a series of weights summing up to 1 that indicate
            the weight of each `h` in contributing to the randomized
            predictor, or a callable corresponding to a deterministic
            `predict` function.
        lambda_vec : pandas.Series
            lambda vector

        Returns
        -------
        tuple
            tuple `(L, L_high, gamma, error)` where `L` is the value of the
            Lagrangian, `L_high` is the value of the Lagrangian under the best
            response of the lambda player, `gamma` is the vector of constraint
            violations, and `error` is the empirical error
        """
        if callable(Q):
            error = self.obj.gamma(Q)[0]
            gamma = self.constraints.gamma(Q)
        else:
            error = self.errors[Q.index].dot(Q)
            gamma = self.gammas[Q.index].dot(Q)

        if self.opt_lambda:
            lambda_projected = self.constraints.project_lambda(lambda_vec)
            L = error + np.sum(lambda_projected * (gamma - self.constraints.bound()))
        else:
            L = error + np.sum(lambda_vec * (gamma - self.constraints.bound()))

        max_constraint = (gamma - self.constraints.bound()).max()
        L_high = error if max_constraint <= 0 else error + self.B * max_constraint
        return L, L_high, gamma, error

    def eval_gap(self, Q, lambda_hat, nu):
        r"""Return the duality gap object for the given :math:`Q` and :math:`\hat{\lambda}`."""
        L, L_high, gamma, error = self._eval(Q, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul * lambda_hat)
            L_low_mul, _, _, _ = self._eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if L_low_mul < result.L_low:
                result.L_low = L_low_mul
            if result.gap() > nu + PRECISION:
                break
        return result

    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (
                self.gammas.sub(self.constraints.bound(), axis=0),
                -np.ones((n_constraints, 1)),
            ),
            axis=1,
        )
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="simplex")
        Q = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]
        result_dual = opt.linprog(dual_c, A_ub=dual_A_ub, b_ub=dual_b_ub, bounds=dual_bounds, method="simplex")
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (Q, lambda_vec, self.eval_gap(Q, lambda_vec, nu))
        return self.last_linprog_result

    def _call_oracle(self, lambda_vec):
        signed_weights = self.obj.signed_weights() + self.constraints.signed_weights(lambda_vec)
        if self.constraints.PROBLEM_TYPE == "classification":
            y = 1 * (signed_weights > 0)
        else:
            y = self.constraints.y_as_series

        w = signed_weights.abs()
        sample_weight = self.constraints.total_samples * w / w.sum()

        estimator = clone(estimator=self.estimator, safe=False)
        estimator.fit(self.constraints.X, y, sample_weight=sample_weight)
        return estimator

    def best_h(self, lambda_vec):
        """
        Solve the best-response problem.

        Description
        -----------
        Returns the classifier that solves the best-response problem for
        the vector of Lagrange multipliers `lambda_vec`.
        """
        classifier = self._call_oracle(lambda_vec)

        def h(X):
            pred = classifier.predict(X)
            # Some estimators return an output of the shape (num_preds, 1) - flatten such
            # results
            if getattr(pred, "flatten", None) is not None:
                pred = pred.flatten()
            return pred

        h_error = self.obj.gamma(h)[0]
        h_gamma = self.constraints.gamma(h)
        h_value = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            values = self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = values.idxmin()
            best_value = values[best_idx]
        else:
            best_idx = -1
            best_value = np.inf

        if h_value < best_value - PRECISION:
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.predictors.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            self.lambdas[h_idx] = lambda_vec.copy()
            best_idx = h_idx

        return self.hs[best_idx], best_idx


class _GapResult:
    """The result of a duality gap computation."""

    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L - self.L_low, self.L_high - self.L)
