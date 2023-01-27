# Base Imports
import numpy as np
import pandas as pd
from sklearn import metrics

from ._validation import _array_like_to_numpy


def extract_columns(df, cols):
    """
    From pandas df to numpy arrays

    Description
    ----------
    This function returns numpy arrays for each column in cols

    Parameters
    ----------
    df : pandas DataFrame
        The data
    cols : list of str
        The column names

    Returns
    -------
    list of numpy arrays
    """
    out = []
    for c in cols:
        out.append(df[c].to_numpy())
    return out


def extract_group_vectors(p_attr, groups):
    """
    Extract Group Vectors

    Description
    ----------
    This function returns a list of group membership vectors
    (one for each group).

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    groups : list of str
        Group names

    Returns
    -------
    list of membership vectors
    """
    p_attr = _array_like_to_numpy(p_attr)
    out = []
    for g in groups:
        out.append(p_attr == g)
    return out


def mat_to_binary(mat, top=None, thresh=0.5):
    """
    Formatting helper function - makes a matrix binary

    Description
    ----------
    Given a recommender matrix, it will coerce to the binary form
    (shown / not shown). This will be first according to the top k rule
    if top is not None. If top is None, it will threshold scores at
    thresh (by default 0.5).

    Parameters
    ----------
    mat : numpy ndarray
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each item.
    top : int
        If not None, the top k scores that are shown to each user
    thresh : float
        Float in (0,1) range indicating threshold at which
        a given item is shown to user

    Returns
    -------
    float
        Binary matrix
    """
    # Case 1 : it's binary
    if np.array_equal(mat, mat.astype(bool)):
        return mat

    # Case 2 : top is not None
    if top is not None:
        # preprocess
        mat = np.nan_to_num(mat, copy=True, nan=-np.Inf)
        n_users, n_items = mat.shape
        top_items = np.argsort(mat, axis=1)

        # find indices of top items for each user
        columns = top_items[:, -top:].flatten()
        rows = np.arange(0, n_users).repeat(top, axis=0)

        # create
        n_mat = np.zeros((n_users, n_items))
        n_mat[rows, columns] = 1

        # impute nans with 0
        n_mat[np.isnan(n_mat)] = 0

    # Case 3 : the score is thresholded
    else:
        n_mat = mat >= thresh

    return n_mat * 1


def normalize_tensor(tensor):
    """
    Formatting helper function - normalises a tensor to [0,1] range

    Description
    ----------
    Given a numpy ndarray, we return a normalised array so that all
    values are in [0,1] range

    Parameters
    ----------
    tensor : numpy ndarray
        a numpy ndarray

    Returns
    -------
    numpy ndarray
        Normalised tensor
    """
    a = np.nanmin(tensor)
    b = np.nanmax(tensor)
    return (tensor - a) / (b - a)


def slice_arrays_by_quantile(q, scores, arr_ls):
    """
    Slice arrays by quantile

    Description
    ----------
    Slices to above given quantile for list of arrays.

    Parameters
    ----------
    q : numpy array
        Quantile
    scores : numpy array
        The scores to slice according to the quantile q
    arr_ls :
        The list of arrays to slice

    Returns
    -------
    list of numpy arrays
        sliced_arrays
    """

    # Get the top indices according to quantile
    scores = np.array(scores)
    top = np.quantile(scores, q)
    top_ind = scores.reshape(1, -1) >= top.reshape(-1, 1)

    sliced_arrays = [[arr[i] for i in top_ind] for arr in arr_ls]

    return sliced_arrays


def recommender_formatter(
    df, users_col, groups_col, items_col, scores_col, aggfunc="sum"
):
    """
    Recommender formatter

    Description
    ----------
    Coerce pandas (users-groups-items-scores) dataframe to format we use
    for recommender bias metrics: pivoted matrix of scores with shape
    (num_users, num_items).

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with recommender data
    users_col : str
        The name of the colum with user id's
    groups_col : str
        The name of the column with group information
    items_col : str
        The name of the column with item id's
    scores_col : str
        The name of the column with score
        (predicted or true) of user on given item
    aggfunc : 'sum' or 'mean'
        the aggregation function for duplicate index, column pairs

    Returns
    -------
    list of numpy ndarray
        df_pivot, p_attr
    """
    # pivot dataframe on users and items
    df_pivot = df.pivot_table(
        index=users_col, columns=items_col, values=scores_col, aggfunc=aggfunc
    )
    # we need to get group info for each user
    user_to_group = dict(zip(df[users_col], df[groups_col]))
    p_attr = np.array(df_pivot.index.map(user_to_group))

    return df_pivot, p_attr
