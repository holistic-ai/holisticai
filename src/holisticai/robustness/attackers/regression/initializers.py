import bisect

import numpy as np


def inf_flip(X, y, proportion):
    """
    Infinitesimal flip sampling.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values.
    proportion : float
        The proportion of points to flip.

    Returns
    -------
    np.ndarray
        The flipped input samples.
    np.ndarray
        The flipped target values.
    """
    count = int(X.shape[0] * proportion / (1 - proportion) + 0.5)
    X = np.matrix(X)
    y = np.array(y)
    inv_cov = (0.01 * np.eye(X.shape[1]) + np.dot(X.T, X)) ** -1
    H = np.dot(np.dot(X, inv_cov), X.T)
    bests = np.sum(H, axis=1)
    room = 0.5 + np.abs(y - 0.5)
    yvals = 1 - np.floor(0.5 + y)
    stat = np.multiply(bests.ravel(), room.ravel())
    stat = stat.tolist()[0]
    totalprob = sum(stat)
    allprobs = [0]
    poisinds = []
    for i in range(X.shape[0]):
        allprobs.append(allprobs[-1] + stat[i])
    allprobs = allprobs[1:]
    for _ in range(count):
        a = np.random.uniform(low=0, high=totalprob)
        poisinds.append(bisect.bisect_left(allprobs, a))

    return X[poisinds], [yvals[a] for a in poisinds]
