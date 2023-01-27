from typing import Optional

import numpy as np
from tqdm import tqdm

from holisticai.utils.models.recommender._rsbase import RecommenderSystemBase
from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from .common_utils.propensity_utils import popularity_model_propensity
from .common_utils.utils import updateP, updateQ


class PopularityPropensityMF(BMImp, RecommenderSystemBase):
    """
    Popularity Propensity Matrix Factorization can be used for Recommender Systems.
    This model is trained with propensity matrix factorization defined in (Eq. 1) [1]. Here the propensity P(u,i)
    is estimated based on popularity.

    References:
        [1] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and
        Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning
        and evaluation. arXiv preprint arXiv:1602.05352 (2016).
    """

    def __init__(
        self,
        K: Optional[int] = 10,
        beta: Optional[float] = 0.02,
        steps: Optional[int] = 100,
        verbose: Optional[int] = 0,
    ):
        """
        Init Popularity Propensity Matrix Factorization

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
        """
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
        """
        N, M = R.shape
        Q = Q.T
        K = P.shape[1]
        b = self.beta * np.identity(K)
        invprop = 1 / propensity
        u_rated_items = [np.where(R[u, :] > 0)[0] for u in range(N)]
        i_rated_users = [np.where(R[:, i] > 0)[0] for i in range(M)]
        for _ in tqdm(range(self.steps), leave=self.verbose > 0):
            P = updateP(P, b, R, Q, u_rated_items, invprop)
            Q = updateQ(P, b, R, Q, i_rated_users, invprop)
        return P, Q
