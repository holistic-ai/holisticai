import os

import numpy as np
from testing_utils.tests_utils import small_recommender_dataset

from holisticai.bias.mitigation import FairRec

np.random.seed(42)


def test_two_sided_fairness(small_recommender_dataset):
    data_matrix, _ = small_recommender_dataset
    numUsers, _ = data_matrix.shape

    rec_size = 10

    for alpha in np.arange(0, 1, 0.1):
        recommender = FairRec(rec_size, alpha)
        recommender.fit(data_matrix)
        assert len(recommender.recommendation.keys()) == numUsers

        for key in recommender.recommendation.keys():
            assert len(recommender.recommendation[key]) == rec_size
