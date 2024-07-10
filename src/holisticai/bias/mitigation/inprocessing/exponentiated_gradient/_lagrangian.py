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
        self.constraints = constraints
        self.constraints.load_data(X, y, **kwargs)
        self.obj = self.constraints.default_objective()
        self.obj.load_data(X, y, **kwargs)
        self.estimator = estimator
        self.B = B
        self.opt_lambda = opt_lambda
        self.hs = pd.Series(dtype="object")
        self.predictors = pd.Series(dtype="object")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None

    def _eval(self, Q, lambda_vec):
        if callable(Q):
            error = self.obj.gamma(Q)[0]
            gamma = self.constraints.gamma(Q)
        else:
            error = self.errors[Q.index].dot(Q)
            gamma = self.gammas[Q.index].dot(Q)

        lambda_projected = self.constraints.project_lambda(lambda_vec) if self.opt_lambda else lambda_vec
        constraint_violation = gamma - self.constraints.bound()
        L = error + np.sum(lambda_projected * constraint_violation)
        max_constraint = np.max(constraint_violation)
        L_high = error if max_constraint <= 0 else error + self.B * max_constraint
        return L, L_high, gamma, error

    def eval_gap(self, Q, lambda_hat, nu):
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
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result

        linprog_data = self._prepare_linprog_data(n_hs, len(self.constraints.index))
        result = opt.linprog(**linprog_data, method="highs")
        Q = pd.Series(result.x[:-1], self.hs.index)

        dual_linprog_data = self._prepare_dual_linprog_data(linprog_data)
        result_dual = opt.linprog(**dual_linprog_data, method="highs")
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)

        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (Q, lambda_vec, self.eval_gap(Q, lambda_vec, nu))
        return self.last_linprog_result

    def _prepare_linprog_data(self, n_hs, n_constraints):
        c = np.append(self.errors.values, self.B)
        A_ub = np.hstack((self.gammas.sub(self.constraints.bound(), axis=0).values, -np.ones((n_constraints, 1))))
        b_ub = np.zeros(n_constraints)
        A_eq = np.hstack((np.ones((1, n_hs)), np.zeros((1, 1))))
        b_eq = np.ones(1)
        return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq}

    def _prepare_dual_linprog_data(self, linprog_data):
        n_constraints = len(linprog_data["b_ub"])
        dual_c = np.append(linprog_data["b_ub"], -linprog_data["b_eq"])
        dual_A_ub = np.hstack((-linprog_data["A_ub"].T, linprog_data["A_eq"].T))
        dual_b_ub = linprog_data["c"]
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]
        return {"c": dual_c, "A_ub": dual_A_ub, "b_ub": dual_b_ub, "bounds": dual_bounds}

    def _call_oracle(self, lambda_vec):
        signed_weights = self.obj.signed_weights() + self.constraints.signed_weights(lambda_vec)
        y = (
            (signed_weights > 0).astype(int)
            if self.constraints.PROBLEM_TYPE == "classification"
            else self.constraints.y_as_series
        )
        w = np.abs(signed_weights)
        sample_weight = self.constraints.total_samples * w / np.sum(w)
        estimator = clone(self.estimator, safe=False)
        estimator.fit(self.constraints.X, y, sample_weight=sample_weight)
        return estimator

    def best_h(self, lambda_vec):
        classifier = self._call_oracle(lambda_vec)

        def h(X):
            pred = classifier.predict(X)
            return pred.flatten() if hasattr(pred, "flatten") else pred

        h_error = self.obj.gamma(h).iloc[0]
        h_gamma = self.constraints.gamma(h)
        h_value = h_error + np.dot(h_gamma, lambda_vec)

        if not self.hs.empty:
            values = self.errors + np.dot(self.gammas.T, lambda_vec)
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
