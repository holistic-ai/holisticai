import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from tqdm import trange

from holisticai.utils.transformers.bias import SensitiveGroups


class Algorithm:
    def __init__(self, metric, solver, verbose):
        self.metric = metric
        self.solver = solver
        self.verbose = verbose
        self.sens_group = SensitiveGroups()

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
        if r >= 1:
            upper_bound = q + 1
        else:
            upper_bound = q

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
        elif self.metric == "L1":
            norm_p = 1
            d = np.linalg.norm(
                centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1
            )
            w = (-d).reshape(-1)
            return w
        elif self.metric == "L2":
            norm_p = 2
            d = np.linalg.norm(
                centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1
            )
            w = (-d).reshape(-1)
            return w
        else:
            raise Exception(f"Unknown Measure : {self.metric}")

    def compute_cost_function(self, centroids, X, z_pred, z_mod):
        if self.metric == "constant":
            return 1 - np.sum(z_mod * z_pred, axis=0)
        elif self.metric == "L1":
            norm_p = 1
            d = np.linalg.norm(
                centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1
            )
            w = np.mean(d, axis=0) - np.sum(d * z_mod, axis=0)
            return np.sum(w * (1 - np.sum(z_mod * z_pred, axis=0)))
        elif self.metric == "L2":
            norm_p = 2
            d = np.linalg.norm(
                centroids[:, None, :] - X[None, ...], ord=norm_p, axis=-1
            )
            w = np.mean(d, axis=0) - np.sum(d * z_mod, axis=0)
            return np.sum(w * (1 - np.sum(z_mod * z_pred, axis=0)))
        else:
            raise Exception(f"Unknown Measure : {self.metric}")

    def transform(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray,
        centroids: np.ndarray = None,
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
        p = trange(k) if self.verbose > 0 else range(k)
        for i in p:
            Pindex = np.where(P == 1)[0]
            C[i, Pindex + i * n] = 1
            C[i, k * n + i] = 1

        p = trange(k) if self.verbose > 0 else range(k)
        for i in p:
            C[i + k, Pindex + i * n] = -1
            C[i + k, k * n + k + i] = 1

        p = trange(n) if self.verbose > 0 else range(n)
        for i in p:
            C[2 * k + i, i + np.arange(k) * n] = 1

        obj = A
        lhs_eq = C
        rhs_eq = np.concatenate(
            [np.ones(k) * upper_bound, -np.ones(k) * lower_bound, np.ones(n)]
        ).reshape([-1, 1])

        bnd = [(0, 1)] * nvars
        opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method=self.solver)
        new_y_pred = opt.x[: k * n].reshape([k, n]).argmax(axis=0)
        if self.verbose > 0:
            self.cost = self.compute_cost_function(centroids, X, y_pred, new_y_pred)
            print(self.cost)
        return new_y_pred
