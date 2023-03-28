import sys
from typing import Optional

import numpy as np
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax

from holisticai.utils.transformers.bias import BMPreprocessing


def get_xhat_y_hat(prototypes, w, x):
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))), np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    return M, x_hat, y_hat


class ObjectiveFunction:
    def __init__(self, verbose, print_interval):
        self.verbose = verbose
        self.step = 0
        self.print_interval = print_interval

    def log_progress(self, total_loss, L_x, L_y, L_z):
        self.step += 1
        if self.verbose and (self.step % self.print_interval) == 0:
            sys.stdout.write(
                f"step: {self.step}\tloss: {total_loss:.4f}\tL_x: {L_x:.4f}\tL_y: {L_y:.4f}\tL_z: {L_z:.4f}\n"
            )
            sys.stdout.flush()

    def __call__(
        self, parameters, x_a, x_b, y_a, y_b, k=10, A_x=0.01, A_y=0.1, A_z=0.5
    ):

        features_dim = x_a.shape[1]

        w = parameters[:k]
        prototypes = parameters[k:].reshape((k, features_dim))

        M_b, x_hat_b, y_hat_b = get_xhat_y_hat(prototypes, w, x_b)
        M_a, x_hat_a, y_hat_a = get_xhat_y_hat(prototypes, w, x_a)

        y_hat = np.concatenate([y_hat_b, y_hat_a], axis=0)
        y = np.concatenate([y_b.reshape((-1, 1)), y_a.reshape((-1, 1))], axis=0)

        L_x = np.mean((x_hat_b - x_b) ** 2) + np.mean((x_hat_a - x_a) ** 2)
        L_z = np.mean(abs(np.mean(M_b, axis=0) - np.mean(M_a, axis=0)))
        L_y = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))

        total_loss = A_x * L_x + A_y * L_y + A_z * L_z

        self.log_progress(total_loss, L_x, L_y, L_z)

        return total_loss


class LearningFairRepresentation(BMPreprocessing):
    """
    Learning fair representations finds a latent representation which encodes the data well
    while obfuscates information about protected attributes.

    References
    ----------
        Zemel, Rich, et al. "Learning fair representations."
        International conference on machine learning. PMLR, 2013.
    """

    def __init__(
        self,
        k: Optional[int] = 5,
        Ax: Optional[float] = 0.01,
        Ay: Optional[float] = 1.0,
        Az: Optional[float] = 50.0,
        print_interval: Optional[int] = 250,
        maxiter: Optional[int] = 5000,
        maxfun: Optional[int] = 5000,
        verbose: Optional[int] = 0,
        seed: Optional[int] = None,
    ):
        """
        Learning Fair Representation Preprocessing Bias Mitigator

        Description
        -----------
        Initialize Mitigator class.

        Parameters
        ----------
        k : int
            Number of prototypes.
        Ax : float
            Input recontruction quality term weight.
        Az : float
            Fairness constraint term weight.
        Ay : float
            Output prediction error.
        print_interval : int
            Print optimization objective value every print_interval iterations.
        verbose : int
            If zero, then no output.
        seed : int
            Seed to make `predict` repeatable.
        """
        self.seed = seed
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

        self.w = None
        self.prototypes = None
        self.learned_model = None
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.obj_fun = ObjectiveFunction(verbose=verbose, print_interval=print_interval)
        self.problem_type = "binary"

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit.

        Description
        -----------
        Fit data to learn a fair representation transform.

        Parameters
        ----------
        X : matrix-like
            Input data
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Return
        ------
            Self
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        params = self._load_data(
            X=X,
            y_true=y_true,
            group_a=group_a,
            group_b=group_b,
        )
        y_true = params["y_true"]
        group_a = params["group_a"]
        group_b = params["group_b"]
        X = params["X"]

        _, self.features_dim = np.shape(X)
        parameters_initialization = np.random.uniform(
            size=self.k + self.features_dim * self.k
        )
        parameters_bounds = [(0, 1)] * self.k + [
            (None, None)
        ] * self.features_dim * self.k
        args = (
            X[group_a == 1],
            X[group_b == 1],
            y_true[group_a == 1],
            y_true[group_b == 1],
            self.k,
            self.Ax,
            self.Ay,
            self.Az,
        )

        self.learned_model = optim.fmin_l_bfgs_b(
            self.obj_fun,
            x0=parameters_initialization,
            epsilon=1e-5,
            args=args,
            bounds=parameters_bounds,
            approx_grad=True,
            maxfun=self.maxfun,
            maxiter=self.maxiter,
            disp=0,
        )[0]

        self.w = self.learned_model[: self.k]
        self.prototypes = self.learned_model[self.k :].reshape(
            (self.k, self.features_dim)
        )
        return self

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Transform data

        Description
        -----------
        Transform data to a fair representation

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Return
        ------
            Transformed X
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        _, x_hat_b, _ = get_xhat_y_hat(self.prototypes, self.w, X[group_b == 1])
        _, x_hat_a, _ = get_xhat_y_hat(self.prototypes, self.w, X[group_a == 1])

        new_X = X.copy()
        new_X[group_b == 1] = x_hat_b
        new_X[group_a == 1] = x_hat_a

        return new_X

    def fit_transform(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit and transform

        Description
        ----------
        Fit and transform

        Parameters
        ----------
        X : matrix-like
            Input data
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Return
        ------
            Self
        """
        return self.fit(X, y_true, group_a, group_b).transform(X, group_a, group_b)
