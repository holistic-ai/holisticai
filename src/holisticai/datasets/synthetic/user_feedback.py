import numpy as np


def load_data():
    """
    Simulates a completed rating matrix ground truth, feedback matrix time and split test data.
    """
    random_state = np.random.RandomState(42)
    num_user = 500
    num_item = 500
    epsilon = 0.5
    ratings = np.zeros((num_user, num_item))
    for u in range(num_user):
        au = random_state.normal(3.4, 1, 1)
        bu = random_state.normal(0.5, 0.5, 1)
        for i in range(num_item):
            ti = random_state.normal(0.1, 1, 1)
            eij = random_state.normal(0, 1, 1)

            a = au + bu * ti + epsilon * eij
            ratings[u][i] = max(min(round(a[0]), 5), 1)

    time_range = 40  ## split the ratings into 40 time range
    rated_time = random_state.randint(1, time_range, size=(ratings.shape[0], ratings.shape[1]))

    ## get the unique testing for all different debias strateies
    lower_rank = 27
    upper_rank = 30
    test_mask = (rated_time >= lower_rank) & (rated_time <= upper_rank)
    test_ratings = ratings * test_mask

    return ratings, rated_time, test_ratings
