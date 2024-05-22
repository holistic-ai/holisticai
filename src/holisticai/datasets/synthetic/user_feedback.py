import numpy as np


def load_data():
    """
    Simulates a completed rating matrix ground truth, feedback matrix time and split test data.
    """
    num_user = 500
    num_item = 500
    epsilon = 0.5
    ratings = np.zeros((num_user, num_item))
    for u in range(num_user):
        au = np.random.normal(3.4, 1, 1)
        bu = np.random.normal(0.5, 0.5, 1)
        for i in range(num_item):
            ti = np.random.normal(0.1, 1, 1)
            eij = np.random.normal(0, 1, 1)

            a = au + bu * ti + epsilon * eij
            ratings[u][i] = max(min(round(a[0]), 5), 1)

    Time_range = 40  ## split the ratings into 40 time range
    rated_time = np.random.randint(
        1, Time_range, size=(ratings.shape[0], ratings.shape[1])
    )

    ## get the unique testing for all different debias strateies
    test_mask = (27 <= rated_time) & (rated_time <= 30)
    test_ratings = ratings * test_mask

    return ratings, rated_time, test_ratings
