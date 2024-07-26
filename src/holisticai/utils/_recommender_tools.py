import numpy as np

# scikit
from sklearn.metrics import f1_score, precision_score, recall_score

# Formatting
from holisticai.utils import mat_to_binary


def avg_precision(mat_pred, mat_true, top=None, thresh=0.5):
    """
    Average Precision (recommender)

    Description
    ----------
    We average out the precision of recommender predictions over
    all users

    Parameters
    ----------
    mat_pred : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    mat_true : numpy ndarray
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each item.
    top : int
        If not None, the top k scores that are shown to each user
    thresh : float
        Float in (0,1) range indicating threshold at which
        a given item is shown to user

    Returns
    -------
    float
        Average precision
    """
    # Make matrices binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)
    binary_mat_true = mat_to_binary(mat_true, top=top, thresh=thresh)

    num_users, _ = mat_pred.shape
    vals = np.zeros((num_users,))

    # Sum accuracy over users
    for u in range(num_users):
        pred = binary_mat_pred[u]
        true = binary_mat_true[u]
        vals[u] = precision_score(true, pred)

    # Average
    return np.mean(vals)


def avg_recall(mat_pred, mat_true, top=None, thresh=0.5):
    """
    Average Recall (recommender)

    Description
    ----------
    We average out the recall of recommender predictions over
    all users

    Parameters
    ----------
    mat_pred : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    mat_true : numpy ndarray
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each item.
    top : int
        If not None, the top k scores that are shown to each user
    thresh : float
        Float in (0,1) range indicating threshold at which
        a given item is shown to user

    Returns
    -------
    float
        Average recall
    """
    # Make matrices binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)
    binary_mat_true = mat_to_binary(mat_true, top=top, thresh=thresh)

    num_users, _ = mat_pred.shape
    vals = np.zeros((num_users,))

    # Sum accuracy over users
    for u in range(num_users):
        pred = binary_mat_pred[u]
        true = binary_mat_true[u]
        vals[u] = recall_score(true, pred)

    # Average
    return np.mean(vals)


def avg_f1(mat_pred, mat_true, top=None, thresh=0.5):
    """
    Average f1 (recommender)

    Description
    ----------
    We average out the f1 of recommender predictions over
    all users

    Parameters
    ----------
    mat_pred : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    mat_true : numpy ndarray
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each item.
    top : int
        If not None, the top k scores that are shown to each user
    thresh : float
        Float in (0,1) range indicating threshold at which
        a given item is shown to user

    Returns
    -------
    float
        Average f1
    """
    # Make matrices binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)
    binary_mat_true = mat_to_binary(mat_true, top=top, thresh=thresh)

    num_users, _ = mat_pred.shape
    vals = np.zeros((num_users,))

    # Sum accuracy over users
    for u in range(num_users):
        pred = binary_mat_pred[u]
        true = binary_mat_true[u]
        vals[u] = f1_score(true, pred)

    # Average
    return np.mean(vals)


def recommender_rmse(mat_pred, mat_true):
    """
    Recommender RMSE

    Description
    ----------
    We compute the rmse of recommender predictions over
    all non null scores.

    Parameters
    ----------
    mat_pred : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    mat_true : numpy ndarray
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each item.

    Returns
    -------
    float
        Recommender RMSE
    """
    se_diff = (mat_pred - mat_true) ** 2
    return np.sqrt(np.nanmean(se_diff))


def recommender_mae(mat_pred, mat_true):
    """
    Recommender MAE

    Description
    ----------
    We compute the mae of recommender predictions over
    all non null scores.

    Parameters
    ----------
    mat_pred : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    mat_true : numpy ndarray
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each item.

    Returns
    -------
    float
        Recommender MAE
    """
    abs_diff = np.abs(mat_pred - mat_true)
    return np.nanmean(abs_diff)


def entropy(p, q=None):
    """
    Entropy

    Description
    ----------
    We compute the entropy of a probability or relative
    entropy of two probabilities

    Parameters
    ----------
    p : array
        probability vector
    q : array
        probability vector

    Returns
    -------
    float
        Entropy
    """
    p = np.array(p)
    p = p / np.sum(p)

    if q is None:
        return -np.sum(np.where(p != 0, p * np.log(p), 0))

    q = np.array(q)
    q = q / np.sum(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
