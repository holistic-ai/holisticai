import os
import sys

import numpy as np

sys.path.append(os.getcwd())
np.random.seed(42)


def test_running_matrix_factorization_strategies():
    from holisticai.datasets import load_last_fm
    from holisticai.utils import recommender_formatter

    bunch = load_last_fm()
    lastfm = bunch["frame"]
    lastfm["score"] = 1
    lastfm = lastfm.iloc[:300]
    df_pivot, p_attr = recommender_formatter(
        lastfm,
        users_col="user",
        groups_col="sex",
        items_col="artist",
        scores_col="score",
        aggfunc="mean",
    )
    data_matrix = df_pivot.fillna(0).to_numpy()
    numUsers, numItems = data_matrix.shape
    from holisticai.utils.models.recommender.matrix_factorization.non_negative import (
        NonNegativeMF,
    )

    K = 10
    mf = NonNegativeMF(K=K)
    mf.fit(data_matrix)
    assert mf.pred.shape == (numUsers, numItems)

    from holisticai.bias.mitigation import PopularityPropensityMF

    mf = PopularityPropensityMF(K=K, beta=0.02, steps=3, verbose=1)
    mf.fit(data_matrix)
    assert mf.pred.shape == (numUsers, numItems)

    from holisticai.bias.mitigation import DebiasingLearningMF

    mf = DebiasingLearningMF(
        K=K,
        normalization="Vanilla",
        lamda=0.08,
        metric="mse",
        bias_mode="Regularized",
        seed=1,
    )
    mf.fit(data_matrix)
    assert mf.pred.shape == (numUsers, numItems)
