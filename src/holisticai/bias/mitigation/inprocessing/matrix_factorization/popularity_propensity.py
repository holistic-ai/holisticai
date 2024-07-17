from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.bias.mitigation.inprocessing.matrix_factorization.common_utils.propensity_utils import (
    popularity_model_propensity,
)
from holisticai.bias.mitigation.inprocessing.matrix_factorization.common_utils.utils import updateP, updateQ
from holisticai.utils.models.recommender._rsbase import RecommenderSystemBase
from holisticai.utils.transformers.bias import BMInprocessing as BMImp

logger = logging.getLogger(__name__)


class PopularityPropensityMF(BMImp, RecommenderSystemBase):
    """Popularity Propensity Matrix Factorization

    Popularity Propensity Matrix Factorization address selection biases in recommender systems \
    by using causal inference techniques to provide unbiased performance estimators \
    and improve prediction accuracy. This method estimates the probability (propensity) \
    that a user will rate an item and adjusts the training and evaluation processes accordingly.

    Parameters
    ----------
        K : int
            Specifies the number of dimensions.

        beta : float
            Parameter used to update P and Q.

        steps : int
            Number of iterations.

        verbose : int
            If >0, will show progress percentage.

    References:
        .. [1] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and\
        Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning\
        and evaluation. arXiv preprint arXiv:1602.05352 (2016).
    """

    def __init__(
        self,
        K: Optional[int] = 10,
        beta: Optional[float] = 0.02,
        steps: Optional[int] = 100,
        verbose: Optional[int] = 0,
    ):
        self.K = K
        self.beta = beta
        self.steps = steps
        self.verbose = verbose

    def fit(self, X: Optional[np.ndarray], **kargs):
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
        propensity = popularity_model_propensity(X)
        P, Q = self._coordinate_descent_matrix_factorization(X, P0, Q0, propensity)
        self.pred = np.dot(P, Q)
        self.invP = None
        return self

    def _coordinate_descent_matrix_factorization(self, R, P, Q, propensity):
        """
        Coordinate Descent Matrix Factorization Algorithm

        Parameters
        ----------
        R : matrix-like
            rating matrix, 0 means non-raked cases.

        P : matrix-like
            Initial P matrix (numUsers, K)

        Q : matrix-like
            Initial P matrix (numItems, K)

        propensity : matrix-like
            Propensity matrix (numUsers, numItems)

        Returns
        -------
        tuple
            P matrix, Q matrix
        """
        N, M = R.shape
        Q = Q.T
        K = P.shape[1]
        b = self.beta * np.identity(K)
        invprop = 1 / propensity
        u_rated_items = [np.where(R[u, :] > 0)[0] for u in range(N)]
        i_rated_users = [np.where(R[:, i] > 0)[0] for i in range(M)]

        logger.info("Coordinate Descent Matrix Factorization Algorithm")
        for _ in range(self.steps):
            P = updateP(P, b, R, Q, u_rated_items, invprop)
            Q = updateQ(P, b, R, Q, i_rated_users, invprop)
        return P, Q
