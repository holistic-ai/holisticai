from typing import Optional

import numpy as np
import scipy.optimize as optim
from holisticai.utils.transformers.bias import BMPreprocessing
from scipy.spatial.distance import cdist
from scipy.special import softmax
from tqdm import tqdm


def get_x_hat_y_hat(prototypes, w, x):
    """
    Get x_hat and y_hat

    Parameters
    ----------
    prototypes : array-like
        Prototypes
    w : array-like
        Weights
    x : array-like
        Input data

    Returns
    -------
    array-like
        M
    array-like
        x_hat
    array-like
        y_hat
    """
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.maximum(
        np.minimum(np.matmul(M, w.reshape((-1, 1))), 1.0), np.finfo(float).eps
    )
    return M, x_hat, y_hat


class ObjectiveFunction:
    """
    Objective function for optimization

    Parameters
    ----------
    maxiter : int
        Maximum number of iterations
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
        Input recontruction quality term weight. Default is 0.01
    A_y : float, optional
        Output prediction error. Default is 0.1
    A_z : float, optional
        Fairness constraint term weight. Default is 0.5

    Attributes
    ----------
    verbose : int
        If zero, then no output
    x : array-like
        Input data
    y : array-like
        Target vector

    Methods
    -------
    __call__(parameters)
        Call the objective function
    log_progress(total_loss, L_x, L_y, L_z)
        Log the progress
    """
    def __init__(
        self,
        maxiter,
        features_dim,
        verbose,
        x,
        y,
        m_a,
        m_b,
        k=10,
        A_x=0.01,
        A_y=0.1,
        A_z=0.5,
    ):
        self.verbose = verbose
        self.x = x
        self.y = y
        self.one_minus_y = 1 - y
        self.m_a = m_a
        self.m_b = m_b
        self.k = k
        self.A = np.array([A_x, A_y, A_z])
        self.prototypes_shape = (self.k, features_dim)
        self.progress_bar = tqdm(total=maxiter, desc="Optimization Progress")

    def __call__(self, parameters):
        """
        Call the objective function

        Parameters
        ----------
        parameters : array-like
            Parameters

        Returns
        -------
        float
            Total loss
        """
        w = parameters[: self.k]
        prototypes = parameters[self.k :].reshape(self.prototypes_shape)

        M, x_hat, y_hat = get_x_hat_y_hat(prototypes, w, self.x)

        dx2 = (x_hat - self.x) ** 2
        L_x = np.mean(dx2[self.m_b]) + np.mean(dx2[self.m_a])

        L_z = np.mean(
            np.abs(np.mean(M[self.m_b], axis=0) - np.mean(M[self.m_a], axis=0))
        )

        L_y = -np.mean(self.y * np.log(y_hat) + self.one_minus_y * np.log(1.0 - y_hat))

        L = np.array([L_x, L_y, L_z])

        total_loss = np.dot(self.A, L)

        if self.verbose > 0:
            self.log_progress(total_loss, L_x, L_y, L_z)

        return total_loss

    def log_progress(self, total_loss, L_x, L_y, L_z):
        """
        Log the progress

        Parameters
        ----------
        total_loss : float
            Total loss
        L_x : float
            L_x
        L_y : float
            L_y
        L_z : float
            L_z
        """
        self.progress_bar.update()
        self.progress_bar.set_postfix_str(
            f"loss: {total_loss:.3f} L_x: {L_x:.3f} L_y: {L_y:.3f} L_z: {L_z:.3f}"
        )


class LearningFairRepresentation(BMPreprocessing):
    """
    Learning fair representations finds a latent representation which encodes the data well\
    while obfuscates information about protected attributes.

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

    Attributes
    ----------
    seed : int
        Seed to make `predict` repeatable
    k : int
        Number of prototypes
    Ax : float
        Input recontruction quality term weight
    Ay : float
        Output prediction error
    Az : float
        Fairness constraint term weight
    w : array-like
        Weights
    prototypes : array-like
        Prototypes
    learned_model : array-like
        Learned model
    maxiter : int
        Maximum number of iterations
    maxfun : int
        Maximum number of function evaluations
    problem_type : str
        Problem type
    verbose : int
        If zero, then no output

    Methods
    -------
    fit(X, y_true, group_a, group_b)
        Fit data to learn a fair representation transform
    transform(X, group_a, group_b)
        Transform data to a fair representation
    fit_transform(X, y_true, group_a, group_b)
        Fit and transform

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

        Return
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
        X = params["X"]

        features_dim = np.shape(X)[1]

        parameters_initialization = np.random.uniform(
            size=self.k + features_dim * self.k
        )
        parameters_bounds = [(0, 1)] * self.k + [(None, None)] * features_dim * self.k
        args = (
            X,
            y,
            group_a == 1,
            group_b == 1,
            self.k,
            self.Ax,
            self.Ay,
            self.Az,
        )

        obj_fun = ObjectiveFunction(self.maxiter, features_dim, self.verbose, *args)

        self.learned_model = optim.fmin_l_bfgs_b(
            obj_fun,
            x0=parameters_initialization,
            epsilon=1e-5,
            bounds=parameters_bounds,
            approx_grad=True,
            maxfun=self.maxfun,
            maxiter=self.maxiter,
            disp=0,
        )[0]

        self.w = self.learned_model[: self.k]
        self.prototypes = self.learned_model[self.k :].reshape((self.k, features_dim))
        return self

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Transform data
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
        array-like
            Transformed data
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        _, x_hat_b, _ = get_x_hat_y_hat(self.prototypes, self.w, X[group_b == 1])
        _, x_hat_a, _ = get_x_hat_y_hat(self.prototypes, self.w, X[group_a == 1])

        new_X = X.copy()
        new_X[group_b == 1] = x_hat_b
        new_X[group_a == 1] = x_hat_a

        return new_X

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

        Return
        ------
            Self
        """
        return self.fit(X, y, group_a, group_b).transform(X, group_a, group_b)
