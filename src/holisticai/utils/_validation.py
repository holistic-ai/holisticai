import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_same_shape(list_of_arr, names=""):
    """
    Check same shape

    Description
    ----------
    This function checks if all numpy arrays
    in a list have same length

    Parameters
    ----------
    list_of_arr : list of numpy arrays
        input
    names : str
        the name of the inputs

    Returns
    -------
    ValueError or None
    """
    num_dims = len(list_of_arr[0].shape)
    for i in range(num_dims):
        try:
            n = len(np.unique([x.shape[i] for x in list_of_arr]))
            if n > 1:
                raise ValueError(names + " do not all have the same shape.")
        except ValueError as e:
            raise ValueError(names + " do not all have the same shape.") from e


def _check_non_empty(arr, name="", quantile=0):
    """
    Check non empty

    Description
    ----------
    This function checks if a numpy array has
    a sum of 0.

    Parameters
    ----------
    arr : binary array
        input
    name : str
        the name of the input
    quantile : float
        quantile

    Returns
    -------
    warning or None
    """
    num_a = arr.sum()
    if num_a == 0:
        logger.warning(f"{name} has no members at quantile {quantile}")


def _check_classes_input(classes, y_pred, y_true=None):
    """
    Check classes input (multiclass)

    Description
    ----------
    This function checks the input classes is
    composed of the unique values of y_pred (and y_true)

    Parameters
    ----------
    classes : numpy array
        Class vector (categorical)
    y_pred : numpy array
        Predictions vector (categorical)
    y_true : numpy array
        Target vector (categorical)

    Returns
    -------
    ValueError or None
    """
    # case 1 : y_true is None
    if y_true is None:
        _classes = set(classes)
        __classes = set(y_pred)
        if _classes != __classes:
            msg = "classes is not a reordering of unique values in y_pred."
            raise ValueError(msg)
    # case 2
    else:
        _classes = set(classes)
        __classes = set(y_pred).union(set(y_true))
        if _classes != __classes:
            msg = "classes is not a reordering of unique values in y_pred or y_true."
            raise ValueError(msg)


def _check_groups_input(groups, p_attr):
    """
    Check groups input (multiclass)

    Description
    ----------
    This function checks the groups input is
    composed of the unique values of p_attr

    Parameters
    ----------
    groups : numpy array
        Class vector (categorical)
    p_attr : numpy array
        Predictions vector (categorical)

    Returns
    -------
    ValueError or None
    """
    # compare sets
    _groups = set(groups)
    __groups = set(p_attr)
    if _groups != __groups:
        msg = "groups is not a reordering of unique values in p_attr."
        raise ValueError(msg)


def _check_binary(arr, name=""):
    """
    Check binary

    Description
    ----------
    This function checks if a numpy array
    is binary.

    Parameters
    ----------
    arr : numpy array
        input
    name : str
        the name of the input

    Returns
    -------
    ValueError or None
    """
    if not np.array_equal(arr, arr.astype(bool)):
        raise ValueError(name + " is not a binary vector.")


def _check_nan(arr, name=""):
    """
    Check for nan

    Description
    ----------
    This function checks if a numpy array
    has a nan.

    Parameters
    ----------
    arr : numpy array
        input
    name : str
        the name of the input

    Returns
    -------
    ValueError or None
    """
    if pd.isnull(pd.Series(arr)).any():
        raise ValueError(name + " has NaN values.")


def _check_nan_mat(mat, name=""):
    """
    Check for nan

    Description
    ----------
    This function checks if a numpy ndarray
    has a nan.

    Parameters
    ----------
    mat : numpy ndarray
        input
    name : str
        the name of the input

    Returns
    -------
    ValueError or None
    """
    if np.isnan(mat).any():
        return ValueError(name + " has NaN values.")
    return None


def _array_like_to_numpy(arr, name=""):
    """
    Coerce input to numpy (if possible)

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.

    Parameters
    ----------
    arr : array-like
        Input to coerce

    Returns
    -------
    numpy array or TypeError
    """
    try:
        out = np.squeeze(np.asarray(arr))
        if len(out.shape) == 1:
            return out
        raise ValueError
    except TypeError as e:
        msg = f"input {name} is not array-like. This includes numpy 1d arrays, lists,\
            pandas Series or pandas 1d DataFrame"
        raise TypeError(msg) from e


def _matrix_like_to_numpy(arr, name=""):
    """
    Coerce input to numpy (if possible)

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.

    Parameters
    ----------
    arr : matrix-like
        Input to coerce

    Returns
    -------
    numpy array or TypeError
    """
    num_dimensions = 2
    try:
        out = np.squeeze(np.asarray(arr))
        if len(out.shape) == num_dimensions:
            return out
        raise ValueError
    except TypeError as e:
        msg = f"input {name} is not matrix-like. This includes numpy 2d arrays, list of lists \
                or pandas 2d DataFrame"
        raise TypeError(msg) from e


def _matrix_like_to_dataframe(arr, name=""):  # noqa: ARG001
    num_dimensions = 2
    try:
        out = np.squeeze(np.asarray(arr))
        if len(out.shape) == num_dimensions:
            columns = [f"Feature {f}" for f in range(out.shape[1])]
            return pd.DataFrame(out, columns=columns)
        ValueError("input is not matrix-like.")
    except TypeError as e:
        raise TypeError("input is not matrix-like.") from e


def _array_like_to_series(arr, name=""):  # noqa: ARG001
    num_dimensions = 1

    def raise_value_error():
        raise ValueError("input is not array-like.")

    try:
        out = np.squeeze(np.asarray(arr))
        if len(out.shape) == num_dimensions:
            return pd.Series(out)
        raise_value_error()
    except TypeError as e:
        raise TypeError("input is not array-like.") from e


def _coerce_and_check_arr(arr, name="input"):
    """
    Coerce and check array-like

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.
    Also checks for nan values.

    Parameters
    ----------
    arr : array-like
        Input to coerce
    name : str
        The name of array

    Returns
    -------
    numpy array or TypeError
    """
    # coerce to numpy if possible
    np_arr = _array_like_to_numpy(arr, name=name)
    # check for nan values
    _check_nan(np_arr, name=name)
    # return
    return np_arr


def _coerce_and_check_mat(mat, name="input"):
    """
    Coerce and check matrix-like

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.
    Also checks for nan values.

    Parameters
    ----------
    mat : matrix-like
        Input to coerce
    name : str
        The name of array

    Returns
    -------
    numpy ndarray or TypeError
    """
    # coerce to numpy if possible
    np_mat = _matrix_like_to_numpy(mat, name=name)
    # check for nan values
    _check_nan_mat(np_mat, name=name)
    # return
    return np_mat


def _coerce_and_check_binary_group(arr, name=""):
    """
    Coerce and check binary vector

    Description
    ----------
    This function coerces to numpy where
    possible, and return an error if not.
    Also checks vector is binary.

    Parameters
    ----------
    arr : array-like
        Input to coerce
    name : str
        The name of array

    Returns
    -------
    numpy ndarray or TypeError
    """
    # coerce to numpy if possible
    np_arr = _array_like_to_numpy(arr, name=name)
    # check binary (also checks for nans)
    _check_binary(np_arr, name=name)
    # return
    return np_arr


def _classification_checks(group_a, group_b, y_pred, y_true=None):
    """
    Classification checks

    Description
    ----------
    This function checks inputs to
    a classification task

    Returns
    -------
    coerced inputs
    """
    group_a = _coerce_and_check_binary_group(group_a, name="group_a")
    group_b = _coerce_and_check_binary_group(group_b, name="group_b")
    y_pred = _coerce_and_check_binary_group(y_pred, name="y_pred")

    if y_true is None:
        _check_same_shape([group_a, group_b, y_pred], names="group_a, group_b, y_pred")
    else:
        y_true = _coerce_and_check_binary_group(y_true, name="y_true")
        _check_same_shape([group_a, group_b, y_pred, y_true], names="group_a, group_b, y_pred, y_true")

    return group_a, group_b, y_pred, y_true


def _regression_checks(group_a, group_b, y_pred, y_true=None, q=None):
    """
    Regression checks

    Description
    ----------
    This function checks inputs to
    a regression task

    Returns
    -------
    coerced inputs
    """
    group_a = _coerce_and_check_binary_group(group_a, name="group_a")
    group_b = _coerce_and_check_binary_group(group_b, name="group_b")
    y_pred = _coerce_and_check_arr(y_pred, name="y_pred")

    if y_true is None:
        _check_same_shape([group_a, group_b, y_pred], names="group_a, group_b, y_pred")
    else:
        y_true = _coerce_and_check_arr(y_true, name="y_true")
        _check_same_shape([group_a, group_b, y_pred, y_true], names="group_a, group_b, y_pred, y_true")

    if q is not None:
        q = np.atleast_1d(np.array(q).squeeze())
        if len(q.shape) > 1:
            msg = "q should be float or array-like"
            raise ValueError(msg)

    return group_a, group_b, y_pred, y_true, q


def _clustering_checks(group_a, group_b, y_pred, y_true=None, data=None, centroids=None):
    """
    Clustering checks

    Description
    ----------
    This function checks inputs to
    a clustering task

    Returns
    -------
    coerced inputs
    """
    group_a = _coerce_and_check_binary_group(group_a, name="group_a")
    group_b = _coerce_and_check_binary_group(group_b, name="group_b")

    if y_pred is not None:
        y_pred = _coerce_and_check_arr(y_pred, name="y_pred")
        _check_same_shape([group_a, group_b, y_pred], names="group_a, group_b, y_pred")

    if y_true is not None:
        y_true = _coerce_and_check_arr(y_true, name="y_true")
        _check_same_shape([y_pred, y_true], names="y_pred, y_true")

    if data is not None:
        data = _coerce_and_check_mat(data, name="data")

    if centroids is not None:
        centroids = _coerce_and_check_mat(centroids, name="centroids")

    if centroids is not None and data is not None and data.shape[1] != centroids.shape[1]:
        msg = "data and centroids datapoints do not have same dimensionality"
        raise ValueError(msg)

    return group_a, group_b, y_pred, y_true, data, centroids


def _recommender_checks(
    group_a=None,
    group_b=None,
    mat_pred=None,
    mat_true=None,
    top=None,
    thresh=None,
    normalize=None,
):
    """
    Recommender checks

    Description
    ----------
    This function checks inputs to
    a recommender task

    Returns
    -------
    coerced inputs
    """
    if group_a is not None:
        group_a = _coerce_and_check_binary_group(group_a, name="group_a")

    if group_b is not None:
        group_b = _coerce_and_check_binary_group(group_b, name="group_b")
        _check_same_shape([group_a, group_b], names="group_a, group_b")

    if mat_pred is not None:
        mat_pred = _matrix_like_to_numpy(mat_pred, name="mat_pred")

    if mat_true is not None:
        mat_true = _matrix_like_to_numpy(mat_true, name="mat_true")
        _check_same_shape([mat_pred, mat_true], names="mat_pred, mat_true")

    if top is not None and not isinstance(top, int):
        msg = "top has to be an integer"
        raise ValueError(msg)

    if normalize is not None:
        normalize = bool(normalize)

    return group_a, group_b, mat_pred, mat_true, top, thresh, normalize


def _multiclass_checks(p_attr=None, y_pred=None, y_true=None, groups=None, classes=None):
    """
    Multiclass checks

    Description
    ----------
    This function checks inputs to
    a multiclass task

    Returns
    -------
    coerced inputs
    """
    if p_attr is not None:
        p_attr = _coerce_and_check_arr(p_attr, name="p_attr")

    if y_pred is not None:
        y_pred = _coerce_and_check_arr(y_pred, name="y_pred")

    if y_true is not None:
        y_true = _coerce_and_check_arr(y_true, name="y_true")

    # length check
    if p_attr is not None and y_pred is not None:
        _check_same_shape([p_attr, y_pred], names="p_attr, y_pred")

    # length check
    if y_pred is not None and y_true is not None:
        _check_same_shape([y_pred, y_true], names="y_pred, y_true")

    # define groups if not defined (where possible)
    if groups is None:
        groups = np.sort(np.unique(p_attr)) if p_attr is not None else None
    else:
        _check_groups_input(groups, p_attr)

    # define classes if not defined (where possible)
    if classes is None:
        if y_true is not None and y_pred is not None:
            classes = np.sort(np.unique(np.concatenate((y_pred, y_true))))
        elif y_true is None and y_pred is not None:
            classes = np.sort(np.unique(y_pred))
        else:
            classes = None
    elif y_true is not None and y_pred is not None:
        _check_classes_input(classes, y_pred, y_true)
    elif y_true is None and y_pred is not None:
        _check_classes_input(classes, y_pred)
    else:
        classes = None

    return p_attr, y_pred, y_true, groups, classes


def _check_valid_y_proba(y_proba: np.ndarray):
    atol = 1e-3
    threshold = 2
    y_shape = np.shape(y_proba)
    assert len(y_shape) == threshold, f"""y_proba must be 2d tensor, found: {y_shape}"""

    sum_all_probs = np.sum(y_proba, axis=1)
    assert np.all(
        np.isclose(sum_all_probs, 1, atol=atol)
    ), f"""probability must sum to 1 across the possible classes, found: {sum_all_probs}"""

    correct_proba_values = np.all(y_proba <= 1) and np.all(y_proba >= 0)
    assert correct_proba_values, f"""probability values must be in the interval [0,1], found: {y_proba}"""


def _check_numerical_dataframe(df: pd.DataFrame):
    """
    Check numerical DataFrame

    Description
    ----------
    This function checks if a dataframe is numerical
    or can be converted to numerical.

    Parameters
    ----------
    df : pandas DataFrame
        input

    Returns
    -------
    ValueError or converted DataFrame
    """
    try:
        return df.astype(float)
    except ValueError as e:
        raise ValueError("DataFrame cannot be converted to numerical values") from e


def _check_columns(df: pd.DataFrame, column: str):
    """
    Check columns

    Description
    ----------
    This function checks if a column exists in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        input
    column : str
        column name

    Returns
    -------
    ValueError or None
    """
    if column not in df.columns:
        msg = f"Column '{column}' does not exist in DataFrame"
        raise ValueError(msg)
