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


def adaptive(X_tr, Y_tr, proportion):
    """
    Generate poisoned data for regression tasks using an adaptive attack strategy.

    Parameters
    ----------
    X_tr : array-like of shape (n_samples, n_features)
        The training input samples.
    Y_tr : array-like of shape (n_samples,)
        The target values (class labels) as integers or floats.
    proportion : float
        The proportion of poisoned data to generate.

    Returns
    -------
    X_pois : numpy.matrix
        The generated poisoned input samples.
    Y_pois : list
        The generated poisoned target values.

    Notes
    -----
    This function creates poisoned data by iteratively selecting samples based on
    their influence on the model and modifying their labels. The number of poisoned
    samples is determined by the specified proportion.
    """
    X_tr_copy = np.copy(X_tr)
    Y_tr_copy = np.array(Y_tr)
    count = int(X_tr.shape[0] * proportion / (1 - proportion) + 0.5)
    yvals = 1 - np.floor(0.5 + Y_tr_copy)
    diff = (yvals - Y_tr_copy).ravel()
    X_pois = np.zeros((count, X_tr.shape[1]))
    Y_pois = []
    for i in range(count):
        inv_cov = np.linalg.inv(0.01 * np.eye(X_tr_copy.shape[1]) + np.dot(X_tr_copy.T, X_tr_copy))
        H = np.dot(np.dot(X_tr_copy, inv_cov), X_tr_copy.T)
        bests = np.sum(H, axis=1)
        stat = np.multiply(bests.ravel(), diff)
        indtoadd = np.random.choice(stat.shape[0], p=np.abs(stat) / np.sum(np.abs(stat)))
        X_pois[i] = X_tr_copy[indtoadd, :]
        X_tr_copy = np.delete(X_tr_copy, indtoadd, axis=0)
        diff = np.delete(diff, indtoadd, axis=0)
        Y_pois.append(yvals[indtoadd])
        yvals = np.delete(yvals, indtoadd, axis=0)
    return np.matrix(X_pois), Y_pois


def randflip(X_tr, Y_tr, proportion):
    """
    Randomly flips a proportion of the labels in the training data.

    Parameters
    ----------
    X_tr : array-like, shape (n_samples, n_features)
        The training input samples.
    Y_tr : array-like, shape (n_samples,)
        The training labels.
    proportion : float
        The proportion of labels to flip. Must be between 0 and 1.

    Returns
    -------
    X_pois : numpy.matrix, shape (count, n_features)
        The input samples corresponding to the flipped labels.
    Y_pois : list, shape (count,)
        The flipped labels.
    """
    count = int(X_tr.shape[0] * proportion / (1 - proportion) + 0.5)
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    Y_pois = [1 if 1 - Y_tr[i] > 0.5 else 0 for i in poisinds]
    return np.matrix(X_tr)[poisinds], Y_pois


def randflipnobd(X_tr, Y_tr, proportion):
    """
    Randomly flips the labels of a proportion of the training data.

    Parameters
    ----------
    X_tr : array-like of shape (n_samples, n_features)
        The training input samples.
    Y_tr : array-like of shape (n_samples,)
        The training labels.
    proportion : float
        The proportion of the training data to flip. Must be between 0 and 1.

    Returns
    -------
    X_pois : numpy.matrix
        The subset of training input samples whose labels were flipped.
    Y_pois : list
        The flipped labels corresponding to `X_pois`.

    Notes
    -----
    The function randomly selects a subset of the training data and flips their labels.
    The number of samples to flip is determined by the given proportion.
    """
    count = int(X_tr.shape[0] * proportion / (1 - proportion) + 0.5)
    poisinds = np.random.choice(X_tr.shape[0], count, replace=False)
    Y_pois = [1 - Y_tr[i] for i in poisinds]
    return np.matrix(X_tr)[poisinds], Y_pois
