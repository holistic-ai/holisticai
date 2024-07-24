from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from holisticai.utils.transformers.bias import BMPreprocessing
from jax.nn import softmax
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def get_x_hat_y_hat(prototypes, w, x):
    m = softmax(-jnp.linalg.norm(x[:, None] - prototypes, axis=2), axis=1)
    x_hat = jnp.dot(m, prototypes)
    y_hat = jnp.clip(jnp.dot(m, w.reshape((-1, 1))), jnp.finfo(float).eps, 1.0)
    return m, x_hat, y_hat


class ObjectiveFunction:
    def __init__(self, features_dim, verbose, x, y, m_a, m_b, k=10, A_x=0.01, A_y=0.1, A_z=0.5):
        """
        Objective function for optimization

        Parameters
        ----------
        features_dim : int
            Number of features
        verbose : int
            If zero, then no output
        x : array-like
            Input data
        y : array-like
            Target vector
        m_a : array-like
            Mask vector
        m_b : array-like
            Mask vector
        k : int, optional
            Number of prototypes. Default is 10
        A_x : float, optional
            Input reconstruction quality term weight. Default is 0.01
        A_y : float, optional
            Output prediction error. Default is 0.1
        A_z : float, optional
            Fairness constraint term weight. Default is 0.5
        """
        self.verbose = verbose
        self.x = x
        self.y = y
        self.one_minus_y = 1 - y
        self.m_a = m_a
        self.m_b = m_b
        self.k = k
        self.A = jnp.array([A_x, A_y, A_z])
        self.prototypes_shape = (self.k, features_dim)

    def __call__(self, parameters):
        w = parameters[: self.k]
        prototypes = parameters[self.k :].reshape(self.prototypes_shape)

        m, x_hat, y_hat = get_x_hat_y_hat(prototypes, w, self.x)

        dx2 = (x_hat - self.x) ** 2
        loss_x = jnp.mean(dx2[self.m_b]) + jnp.mean(dx2[self.m_a])

        loss_z = jnp.mean(jnp.abs(jnp.mean(m[self.m_b], axis=0) - jnp.mean(m[self.m_a], axis=0)))

        loss_y = -jnp.mean(self.y * jnp.log(y_hat) + self.one_minus_y * jnp.log(1.0 - y_hat))

        loss = jnp.array([loss_x, loss_y, loss_z])

        total_loss = jnp.dot(self.A, loss)

        return total_loss


class LearningFairRepresentation(BMPreprocessing):
    """
    Learning fair representations finds a latent representation which encodes the data well\
    while obfuscates information about protected attributes [1].

    Parameters
    ----------
    k : int, optional
        Number of prototypes. Default is 5
    Ax : float, optional
        Input recontruction quality term weight. Default is 0.01
    Ay : float, optional
        Output prediction error. Default is 1.0
    Az : float, optional
        Fairness constraint term weight. Default is 50.0
    maxiter : int, optional
        Maximum number of iterations. Default is 5000
    maxfun : int, optional
        Maximum number of function evaluations. Default is 5000
    verbose : int, optional
        If zero, then no output. Default is 0
    seed : int, optional
        Seed to make `predict` repeatable. Default is None

    References
    ----------
    .. [1] Zemel, Rich, et al. "Learning fair representations."
        International conference on machine learning. PMLR, 2013.
    """

    def __init__(
        self,
        k: Optional[int] = 5,
        Ax: Optional[float] = 0.01,
        Ay: Optional[float] = 1.0,
        Az: Optional[float] = 50.0,
        maxiter: Optional[int] = 5000,
        maxfun: Optional[int] = 5000,
        verbose: Optional[int] = 0,
        seed: Optional[int] = None,
    ):
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
        self.problem_type = "binary"
        self.verbose = verbose

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit data to learn a fair representation transform.

        Parameters
        ----------
        X : matrix-like
            Input data
        y : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
            Self
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        params = self._load_data(
            X=X,
            y=y,
            group_a=group_a,
            group_b=group_b,
        )
        y = params["y"].reshape([-1, 1])
        group_a = params["group_a"]
        group_b = params["group_b"]
        x = params["X"]

        features_dim = np.shape(x)[1]

        parameters_initialization = np.random.uniform(size=self.k + features_dim * self.k)
        parameters_bounds = [(0, 1)] * self.k + [(None, None)] * features_dim * self.k
        args = (
            x,
            y,
            group_a == 1,
            group_b == 1,
            self.k,
            self.Ax,
            self.Ay,
            self.Az,
        )

        obj_fun = ObjectiveFunction(features_dim, self.verbose, *args)

        @jax.jit
        def objective(params):
            return obj_fun(params)

        result = minimize(
            objective,
            parameters_initialization,
            method="L-BFGS-B",
            bounds=parameters_bounds,
            options={"maxiter": self.maxiter, "disp": 0},
        )
        self.learned_model = result.x
        self.w = self.learned_model[: self.k]
        self.prototypes = self.learned_model[self.k :].reshape((self.k, features_dim))
        return self

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Transform data to a fair representation

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
        array-like
            Transformed data
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        x = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        _, x_hat_b, _ = get_x_hat_y_hat(self.prototypes, self.w, x[group_b == 1])
        _, x_hat_a, _ = get_x_hat_y_hat(self.prototypes, self.w, x[group_a == 1])

        new_x = x.copy()
        new_x[group_b == 1] = x_hat_b
        new_x[group_a == 1] = x_hat_a

        return new_x

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit and transform

        Parameters
        ----------
        X : matrix-like
            Input data
        y : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
            Self
        """
        return self.fit(X, y, group_a, group_b).transform(X, group_a, group_b)
