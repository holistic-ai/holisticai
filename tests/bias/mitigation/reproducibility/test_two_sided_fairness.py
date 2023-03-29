import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

np.random.seed(42)


def test_two_sided_fairness():
    from holisticai.datasets import load_last_fm
    from holisticai.utils import recommender_formatter

    bunch = load_last_fm()
    lastfm = bunch["frame"]
    lastfm["score"] = 1
    lastfm = lastfm.iloc[:100]
    df_pivot, p_attr = recommender_formatter(
        lastfm,
        users_col="user",
        groups_col="sex",
        items_col="artist",
        scores_col="score",
        aggfunc="mean",
    )
    data_matrix = df_pivot.fillna(0).to_numpy()
    numUsers, _ = data_matrix.shape

    from holisticai.bias.mitigation import FairRec

    # size of recommendation
    rec_size = 10

    for alpha in np.arange(0, 1, 0.1):
        recommender = FairRec(rec_size, alpha)
        recommender.fit(data_matrix)
        assert len(recommender.recommendation.keys()) == numUsers

        for key in recommender.recommendation.keys():
            assert len(recommender.recommendation[key]) == rec_size
