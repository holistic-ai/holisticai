import os
import sys

import numpy as np

from tests.bias.mitigation.testing_utils.utils import small_recommender_dataset

np.random.seed(42)


def test_running_matrix_factorization_strategies(small_recommender_dataset):
    data_matrix, _ = small_recommender_dataset
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
