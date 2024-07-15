import logging

import numpy as np
from holisticai.utils.transformers.bias import SensitiveGroups
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(self, metric, solver, verbose):
        self.metric = metric
        self.solver = solver
        self.verbose = verbose
        self._sensgroups = SensitiveGroups()

    def init_parameters(self, y_pred: np.ndarray, p_attr: np.ndarray):
        k = len(np.unique(y_pred))

        n_classes = np.max(y_pred) + 1
        y_oh = np.zeros((y_pred.size, n_classes))
        y_oh[np.arange(y_pred.size), y_pred] = 1
        p_attr = np.array(p_attr).reshape(-1)
        Nx = np.sum(p_attr)
        q = Nx // k
        r = Nx - q * k
        lower_bound = q
        upper_bound = q + 1 if r >= 1 else q

        P = p_attr.astype("int32")
        n = len(P)
        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "len_data": n,
            "y_oh": y_oh,
            "P": P,
            "k": k,
        }

    def penalty_weights(self, X=None, centroids: np.ndarray = None):
        if self.metric == "constant":
            return -1

        if self.metric == "L1":
            norm_p = 1
            d = np.linalg.norm(centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1)
            w = (-d).reshape(-1)
            return w

        if self.metric == "L2":
            norm_p = 2
            d = np.linalg.norm(centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1)
            w = (-d).reshape(-1)
            return w

        message = f"Penalty Weights not implemented : {self.metric}"
        raise NotImplementedError(message)

    def compute_cost_function(self, centroids, X, z_pred, z_mod):
        if self.metric == "constant":
            return 1 - np.sum(z_mod * z_pred, axis=0)

        if self.metric == "L1":
            norm_p = 1
            d = np.linalg.norm(centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1)
            w = np.mean(d, axis=0) - np.sum(d * z_mod, axis=0)
            return np.sum(w * (1 - np.sum(z_mod * z_pred, axis=0)))

        if self.metric == "L2":
            norm_p = 2
            d = np.linalg.norm(centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1)
            w = np.mean(d, axis=0) - np.sum(d * z_mod, axis=0)
            return np.sum(w * (1 - np.sum(z_mod * z_pred, axis=0)))

        raise NotImplementedError(f"Cost Function not implemented : {self.metric}")

    def transform(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray,
        centroids: np.ndarray,
    ):
        params = self.init_parameters(y_pred, group)
        n = params["len_data"]
        k = params["k"]
        P = params["P"]
        upper_bound = params["upper_bound"]
        lower_bound = params["lower_bound"]

        weights = self.penalty_weights(X=X, centroids=centroids)
        zi = (-params["y_oh"].T).reshape(-1)
        A = -np.concatenate([weights * zi, np.zeros(2 * k)])

        ncons = 2 * k + n
        nvars = k * n + 2 * k
        C = lil_matrix((ncons, nvars))
        for i in range(k):
            Pindex = np.where(P == 1)[0]
            C[i, Pindex + i * n] = 1
            C[i, k * n + i] = 1

        for i in range(k):
            C[i + k, Pindex + i * n] = -1
            C[i + k, k * n + k + i] = 1

        for i in range(n):
            C[2 * k + i, i + np.arange(k) * n] = 1

        obj = A
        lhs_eq = C
        rhs_eq = np.concatenate([np.ones(k) * upper_bound, -np.ones(k) * lower_bound, np.ones(n)]).reshape([-1, 1])

        bnd = [(0, 1)] * nvars
        opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method=self.solver)
        new_y_pred = opt.x[: k * n].reshape([k, n]).argmax(axis=0)
        if self.verbose > 0:
            self.cost = self.compute_cost_function(centroids, X, y_pred, new_y_pred)
            logger.info(self.cost)
        return new_y_pred
