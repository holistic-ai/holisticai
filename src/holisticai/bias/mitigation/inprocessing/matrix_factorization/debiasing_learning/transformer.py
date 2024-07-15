from __future__ import annotations

from typing import Optional

import numpy as np
import scipy
from holisticai.bias.mitigation.inprocessing.matrix_factorization.debiasing_learning.algorithm import (
    DebiasingLearningAlgorithm,
)
from holisticai.utils.models.recommender._rsbase import RecommenderSystemBase
from holisticai.utils.transformers.bias import BMInprocessing as BMImp


class DebiasingLearningMF(BMImp, RecommenderSystemBase):
    """Debiasing Learning Matrix Factorization

    Debiasing Learning Matrix Factorization handles selection biases by adapting\
    models and estimation techniques from causal inference. The strategy leads to\
    unbiased performance estimators despite biased data, and to a matrix factorization\
    method that provides substantially improved prediction performance on real-world data.

    Parameters
    ----------
        K : int
            Specifies the number of dimensions.

        normalization : str
            Strategy to normalize rating matrix. Avaiables are:
                    - 'Vanilla',
                    - 'SelfNormalized'
                    - 'UserNormalized'
                    - 'ItemNormalized'

        lamda : float
            Model parameter.

        metric: str
            Metric used as cost function.

        clip_val: float
            Propensity Clip Value

        seed: int
            Random Seed

        bias_mode: str
            Bias value using in the model:
            - "None": No bias
            - "Free": Use bias wihtout regularizer in the cost function.
            - "Regularized": Use bias with regularizer in the cost function.

        verbose : int
            If >0, will show progress percentage.

    References
    ----------
        .. [1] Schnabel, Tobias, et al. "Recommendations as treatments: Debiasing learning\
        and evaluation." international conference on machine learning. PMLR, 2016.
    """

    def __init__(
        self,
        K: Optional[int] = 10,
        normalization: Optional[str] = "Vanilla",
        lamda: Optional[float] = 0.02,
        metric: Optional[str] = "mse",
        bias_mode: Optional[str] = None,
        clip_val: Optional[float] = -1,
        seed: Optional[int] = None,
        verbose: Optional[int] = 0,
    ):
        self.seed = seed
        self.normalization = normalization
        self.metric = metric
        self.K = K
        self.clip_val = clip_val
        self.lamda = lamda
        self.bias_mode = bias_mode
        self.verbose = verbose
        self.algorithm = DebiasingLearningAlgorithm(
            K=K,
            normalization=normalization,
            metric=metric,
            lamda=lamda,
            clip_val=clip_val,
            bias_mode=bias_mode,
            verbose=self.verbose,
        )
        # self.params = [lamda, K, clipVal, bias_mode]

    def fit(self, X: Optional[np.ndarray], propensities: Optional[np.ndarray] = None):
        """
        Fit model

        Parameters
        ----------

        X : matrix-like
                rating matrix, 0 means non-raked cases.

        propensities : matrix-like (optional)
                Propensity matrix

        Returns
        -------
            self
        """
        invP = None
        if propensities is not None:
            invP = np.reciprocal(propensities)
            invP = np.ma.array(invP, copy=False, mask=np.ma.getmask(X), fill_value=0, hard_mask=True)

        # Get starting params by SVD
        params0 = self._init_parameters(X)
        model_params = self.algorithm.train(X, invP, params0)
        self.pred = self.algorithm.predict(model_params, bias_mode=self.bias_mode)
        self.invP = invP
        return self

    def _init_parameters(self, partial_observations):
        averageObservedRating = np.ma.mean(partial_observations)
        completeRatings = np.ma.filled(partial_observations.astype(float), averageObservedRating)
        numUsers, numItems = np.shape(partial_observations)
        numUsers = completeRatings.shape[0]
        ncv = (min(numUsers, numItems) + self.K) // 2

        u, s, vt = scipy.sparse.linalg.svds(
            completeRatings,
            k=self.K,
            ncv=ncv,
            tol=1e-7,
            which="LM",
            v0=None,
            maxiter=2000,
            return_singular_vectors=True,
        )
        P = u
        Q = np.transpose(np.multiply(vt, s[:, None]))
        userBias = np.zeros(numUsers)
        itemBias = np.zeros(numItems)
        params0 = (P, Q, userBias, itemBias, averageObservedRating)
        return params0
