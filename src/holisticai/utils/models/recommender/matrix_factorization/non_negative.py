from __future__ import annotations

import numpy as np
from holisticai.utils.models.recommender._rsbase import RecommenderSystemBase
from sklearn.decomposition import NMF


class NonNegativeMF(RecommenderSystemBase):
    """
    This model is trained using conventional matrix factorization (Eq.1 [1] with Pu,i = 1),
    and the system always selects the top predicted item for each user, and adds
    it to the next (new) training set.

    References:
        [1] Sun, Wenlong, et al. "Debiasing the human-recommender system feedback loop in
        collaborative filtering." Companion Proceedings of The 2019 World Wide Web
        Conference. 2019.
    """

    def __init__(self, K: int | None = 10):
        """
        Init Wrapper Non Negative Matrix Factorization

        Parameters
        ----------

        K : int
            Specifies the number of dimensions.
        """
        self.K = K

    def fit(self, X: np.ndarray | None):
        """
        Fit model

        Parameters
        ----------

        X : matrix-like
            rating matrix, 0 means non-raked cases.
        """
        model = NMF(n_components=self.K, init="random", max_iter=200)
        P = model.fit_transform(X)
        Q = model.components_
        self.pred = np.dot(P, Q)
        self.invP = None
        return self
