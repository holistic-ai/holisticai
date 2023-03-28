from typing import Optional

import numpy as np
import pandas as pd

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from .algorithm import FairRecAlg


class FairRec(BMImp):
    """
    FairRecommendationSystem (FairRec), exhibes the desired two-sided fairness by
    mapping the fair recommendation problem to a fair allocation problem; moreover,
    it is agnostic to the specifics of the data-driven model (that estimates the
    product-customer relevance scores) which makes it more scalable and easy to adapt [1].
    References:
        [1] Patro, Gourab K., et al. "Fairrec: Two-sided fairness for personalized
        recommendations in two-sided platforms." Proceedings of The Web Conference 2020. 2020.
    """

    def __init__(
        self, rec_size: Optional[int] = 10, MMS_fraction: Optional[float] = 0.5
    ):
        """
        Init FairRec algorithm
        Parameters
        ----------
        rec_size : int
            Specifies the number of recommended items.
        MMS_fraction : float
            Maximin Share (MMS) threshold of producers exposure.
        """

        self.rec_size = rec_size
        self.MMS_fraction = MMS_fraction

    def fit(self, X):
        algorithm = FairRecAlg(rec_size=self.rec_size, MMS_fraction=self.MMS_fraction)
        self.recommendation = algorithm.rank(X)
        return self

    def predict(self, X: Optional[np.ndarray], top_n: Optional[int] = None):
        """
        Fit model

        Parameters
        ----------
        X : matrix-like
            scored matrix, 0 means non-raked cases.

        Returns
        -------
        recommendations : dict
            A dictionary of recommendations for each user.
        """
        if top_n is None:
            algorithm = FairRecAlg(
                rec_size=self.rec_size, MMS_fraction=self.MMS_fraction
            )
            self.recommendation = algorithm.rank(X)

        dfs = []
        for i, key in enumerate(self.recommendation.keys()):
            df = pd.DataFrame()
            df["Y"] = np.array(self.recommendation[key])
            df["X"] = i
            df["score"] = 1.0
            dfs.append(df)
        dfs = pd.concat(dfs, axis=0)
        return dfs[["X", "Y", "score"]]
