from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.utils.models.recommender._rsbase import RecommenderSystemBase
from holisticai.utils.transformers.bias import BMInprocessing as BMImp

logger = logging.getLogger(__name__)


class BlindSpotAwareMF(BMImp, RecommenderSystemBase):
    """Blind Spot Aware Matrix Factorization

    A blind spot aware Matrix Factorization takes into account the blind spot\
    inherent in the learning phase of the RS. The blind spot size is the\
    number of item with a predicted ratings Ru,i that is smaller than a\
    threshold.

    Parameters
    ----------
        K : int
            Specifies the number of dimensions.

        beta : float
            Parameter used to update P and Q.

        steps : int
            Number of iterations.

        alpha : float
            Model parameter. Alpha is the learning rate.

        lamda : float
            Model parameter. Lambda is the regularization parameter.

        verbose : int
            If >0, will show progress percentage.

    References
    ----------
        .. [1] Sun, Wenlong, et al. "Debiasing the human-recommender system\
        feedback loop in collaborative filtering." Companion Proceedings\
        of The 2019 World Wide Web Conference. 2019.
    """

    def __init__(
        self,
        K: int = 10,
        beta: Optional[float] = 0.002,
        steps: Optional[int] = 200,
        alpha: Optional[float] = 0.002,
        lamda: Optional[float] = 0.2,
        verbose: Optional[int] = 0,
    ):
        self.beta = beta
        self.steps = steps
        self.alpha = alpha
        self.lamda = lamda
        self.K = K
        self.verbose = verbose

    def fit(self, X: np.ndarray, **kargs):
        """
        Fit model

        Parameters
        ----------

        X : matrix-like
            rating matrix, 0 means non-raked cases.

        P0 : matrix-like (optional)
            Initial P matrix (numUsers, K)

        Q0 : matrix-like (optional)
            Initial P matrix (numItems, K)

        Returns
        -------
            self
        """
        numUsers, numItems = X.shape
        P0 = kargs.get("P", np.random.rand(numUsers, self.K))
        Q0 = kargs.get("Q", np.random.rand(numItems, self.K))
        P, Q = self._blindspot_awarness_matrix_factorization(X, P0, Q0)
        self.pred = np.dot(P, Q)
        self.invP = None
        return self

    def _blindspot_awarness_matrix_factorization(self, R, P, Q):
        """
        Blind Spot Awarness Matrix Factorization Algorithm

        Parameters
        ----------
        R : matrix-like
            rating matrix, 0 means non-raked cases.

        P : matrix-like
            Initial P matrix (numUsers, K)

        Q : matrix-like
            Initial P matrix (numItems, K)

        Returns
        -------
        Tuple
            P matrix, Q matrix

        """
        Q = Q.T
        N, M = R.shape
        W = np.ones((N, M))
        i_list, j_list = np.where(R > 0)
        W_s = W[i_list, j_list]

        logger.info("Blind Spot Awarness Matrix Factorization Algorithm Training")
        for _ in range(self.steps):
            for i, j, wij in zip(i_list, j_list, W_s):
                eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                P[i, :] = P[i, :] + self.alpha * (
                    2 * eij * Q[:, j] - self.lamda * P[i, :] - self.beta * (P[i, :] - Q[:, j]) * wij
                )
                Q[:, j] = Q[:, j] + self.alpha * (
                    2 * eij * P[i, :] - self.lamda * Q[:, j] + self.beta * (P[i, :] - Q[:, j]) * wij
                )

        return P, Q
