# Base Imports
import numpy as np
import pandas as pd

# utils
from ...utils._validation import _multiclass_checks


def confusion_matrix(y_pred, y_true, classes=None, normalize=None):
    """
    Confusion Matrix.

    Description
    ----------
    This function computes the confusion matrix. The i,jth
    entry is the number of elements with predicted class i
    and true class j.

    Parameters
    ----------
    y_pred : array-like
        Prediction vector (categorical)
    y_true : array-like
        Target vector (categorical)
    classes (optional) : list
        The unique output classes in order
    normalize (optional) : None, 'pred' or 'class'
        According to which of pred or class we normalize

    Returns
    -------
    numpy ndarray
        Confusion Matrix : shape (num_classes, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import confusion_matrix
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> confusion_matrix(y_pred, y_true, classes=[2, 1, 0])
        2    1    0
    2  1.0  1.0  1.0
    1  0.0  3.0  0.0
    0  1.0  1.0  2.0
    """
    # check and coerce inputs
    _, y_pred, y_true, _, classes = _multiclass_checks(
        p_attr=None,
        y_pred=y_pred,
        y_true=y_true,
        groups=None,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    class_dict = dict(zip(classes, range(num_classes)))

    # initialize the confusion matrix
    confmat = np.zeros((num_classes, num_classes))

    # loop over instances
    for x, y in zip(y_pred, y_true):
        # increment correct entry
        confmat[class_dict[x], class_dict[y]] += 1

    if normalize is None:
        pass

    elif normalize == "pred":
        confmat = confmat / np.sum(confmat, axis=1).reshape(-1, 1)

    elif normalize == "true":
        confmat = confmat / np.sum(confmat, axis=0).reshape(1, -1)

    else:
        raise ValueError('normalize should be one of None, "pred" or "true"')

    return pd.DataFrame(confmat, columns=classes).set_index(np.array(classes))


def confusion_tensor(
    p_attr, y_pred, y_true, groups=None, classes=None, as_tensor=False
):
    """
    Confusion Tensor.

    Description
    ----------
    This function computes the confusion tensor. The k,i,jth
    entry is the number of instances of group k with predicted
    class i and true class j.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector (categorical)
    y_pred : array-like
        Prediction vector (categorical)
    y_true : array-like
        Target vector (categorical)
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    as_tensor (optional) : bool, default False
        Whether we return a tensor or DataFrame

    Returns
    -------
    numpy ndarray
        Confusion Tensor : shape (num_groups, num_classes, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import confusion_tensor
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=True).shape
    (3, 3, 3)
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=True)
    array([[[2., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.]],

        [[0., 0., 1.],
            [0., 2., 0.],
            [1., 0., 0.]],

        [[0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 1.]]])
    >>> confusion_tensor(p_attr, y_pred, y_true, as_tensor=False)
        A	        B	        C
        0	1	2	0	1	2	0	1	2
    0	2.0	0.0	0.0	0.0	0.0	1.0	0.0	1.0	0.0
    1	0.0	1.0	0.0	0.0	2.0	0.0	0.0	0.0	0.0
    2	0.0	1.0	0.0	1.0	0.0	0.0	0.0	0.0	1.0
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)
    class_dict = dict(zip(classes, range(num_classes)))
    group_dict = dict(zip(groups, range(num_groups)))

    # initialize the confusion tensor
    conftens = np.zeros((num_groups, num_classes, num_classes))

    # loop over instances
    for x, y, z in zip(p_attr, y_pred, y_true):
        # increment correct entry
        conftens[group_dict[x], class_dict[y], class_dict[z]] += 1

    # return as a tensor
    if as_tensor is True:
        return conftens

    # return as pandas DataFrame
    elif as_tensor is False:
        d = {}
        for i, group in enumerate(groups):
            # confusion matrix of group number i
            d[group] = pd.DataFrame(conftens[i, :, :], columns=classes).set_index(
                np.array(classes)
            )
        # create a multilevel pandas dataframe
        multi_df = pd.concat(d, axis=1)
        return multi_df

    else:
        raise ValueError("as_tensor should be boolean")


def frequency_matrix(p_attr, y_pred, groups=None, classes=None, normalize="group"):
    """
    Frequency Matrix.

    Description
    ----------
    This function computes the frequency matrix. For each
    group, class pair we compute the count of that group
    for admission within that class. We include the option to normalise
    over groups or classes. By default we normalise by 'group'.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    normalize (optional): None, 'group' or 'class'
        According to which of group or class we normalize

    Returns
    -------
    pandas DataFrame
        Success Rate Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import frequency_matrix
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> frequency_matrix(p_attr, y_pred, normalize='class')
        0     1     2
    A  0.50  0.25  0.25
    B  0.25  0.50  0.25
    C  0.50  0.00  0.50
    """
    # check and coerce inputs
    p_attr, y_pred, _, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)
    class_dict = dict(zip(classes, range(num_classes)))
    group_dict = dict(zip(groups, range(num_groups)))

    # initialize success rate matrix
    sr_mat = np.zeros((num_groups, num_classes))

    # loop over instances
    for x, y in zip(p_attr, y_pred):
        sr_mat[group_dict[x], class_dict[y]] += 1

    # normalize is None, return counts
    if normalize is None:
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    # normalise over rows
    elif normalize == "group":
        sr_mat = sr_mat / sr_mat.sum(axis=1).reshape(-1, 1)
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    # normalise over columns
    elif normalize == "class":
        sr_mat = sr_mat / sr_mat.sum(axis=0).reshape(1, -1)
        return pd.DataFrame(sr_mat, columns=classes).set_index(np.array(groups))

    else:
        raise ValueError("normalize has to be one of None, 'group' or 'class'")


def accuracy_matrix(p_attr, y_pred, y_true, groups=None, classes=None):
    """
    Multiclass Accuracy Matrix.

    Description
    ----------
    Given a protected attribute and multiclass classification task,
    for each group and class this function computes the accuracy of
    predictions on that group for the one vs all classifier of that class.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    pandas DataFrame
        Accuracy Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import accuracy_matrix
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> accuracy_matrix(p_attr, y_pred, y_true)
        0    1    2
    A  1.0  0.5  0.0
    B  0.0  1.0  0.0
    C  0.0  0.0  1.0
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)

    # compute confusion tensor
    conftens = confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)

    # initialize accuracy matrix
    acc_mat = np.zeros((num_groups, num_classes))

    # loop over groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :]
        acc_mat[k, :] = np.diag(confmat_k) / (
            confmat_k.sum(axis=0) + confmat_k.sum(axis=1) - np.diag(confmat_k)
        )

    return pd.DataFrame(acc_mat, columns=classes).set_index(np.array(groups))


def precision_matrix(p_attr, y_pred, y_true, groups=None, classes=None):
    """
    Multiclass Precision Matrix.

    Description
    ----------
    Given a protected attribute and multiclass classification task,
    for each group and class this function computes the precision of
    predictions on that group for the one vs all classifier of that class.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    pandas DataFrame
        Precision Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import precision_matrix
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> precision_matrix(p_attr, y_pred, y_true, groups=None, classes=None)
        0    1    2
    A  1.0  0.5  NaN
    B  0.0  1.0  0.0
    C  NaN  0.0  1.0
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)

    # compute confusion tensor
    conftens = confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)

    # initialize precision matrix
    prec_mat = np.zeros((num_groups, num_classes))

    # loop over groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :]
        prec_mat[k, :] = np.diag(confmat_k) / np.sum(confmat_k, axis=0)

    return pd.DataFrame(prec_mat, columns=classes).set_index(np.array(groups))


def recall_matrix(p_attr, y_pred, y_true, groups=None, classes=None):
    """
    Multiclass Recall Matrix.

    Description
    ----------
    Given a protected attribute and multiclass classification task,
    for each group and class this function computes the recall of
    predictions on that group for the one vs all classifier of that class.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    pandas DataFrame
        Recall Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import recall_matrix
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> recall_matrix(p_attr, y_pred, y_true, groups=None, classes=None)
        0    1    2
    A  1.0  1.0  0.0
    B  0.0  1.0  0.0
    C  0.0  NaN  1.0
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)

    # compute confusion tensor
    conftens = confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)

    # initialize recall matrix
    rec_mat = np.zeros((num_groups, num_classes))

    # loop over groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :]
        rec_mat[k, :] = np.diag(confmat_k) / np.sum(confmat_k, axis=1)

    return pd.DataFrame(rec_mat, columns=classes).set_index(np.array(groups))


def multiclass_equality_of_opp(
    p_attr, y_pred, y_true, groups=None, classes=None, aggregation_fun="mean"
):
    """
    Multiclass Equality of Opportunity.

    Description
    ----------
    This metric is a multiclass generalisation of Equality of
    Opportunity. For each group, compute the matrix of error
    rates (normalised confusion matrix). Compute all distances
    (mean absolute deviation) between such matrices. Then
    aggregate them using the mean, or max strategy.

    Interpretation
    --------------
    The accepted values and bounds for this metric are the same
    as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    scalar
        Multiclass Equality of Opportunity

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import multiclass_equality_of_opp
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> multiclass_equality_of_opp(p_attr, y_pred, y_true)
    0.7962962962962963
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)

    # compute confusion tensor
    conftens = (
        confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)
        + 1e-32
    )
    # normalization constant
    norm = conftens.sum(axis=1)

    # initialize distance matrix
    dist_mat = np.zeros((num_groups, num_groups))

    # distance confusion matrix across groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :] / norm[k, :]
        for j in range(k + 1, num_groups):
            confmat_j = conftens[j, :, :] / norm[j, :]
            dist_mat[k, j] = np.sum(np.abs(confmat_k - confmat_j)) / (2 * num_classes)

    if aggregation_fun == "max":
        res = np.max(dist_mat)
    else:
        res = np.sum(dist_mat) / (num_groups * (num_groups - 1) / 2)

    return res


def multiclass_average_odds(
    p_attr, y_pred, y_true, groups=None, classes=None, aggregation_fun="mean"
):
    """
    Multiclass Average Odds.

    Description
    ----------
    This metric is a multiclass generalisation of Average
    Odds. For each group, compute the matrix of error
    rates (normalised confusion matrix). Average these
    matrices over rows, and compute all pariwise distances
    (mean absolute deviation) between the resulting vectors.
    Aggregate results using either mean or max strategy.

    Interpretation
    --------------
    The accepted values and bounds for this metric are the same
    as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    scalar
        Multiclass Average Odds

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import multiclass_average_odds
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> multiclass_average_odds(p_attr, y_pred, y_true)
    0.16666666666666666
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    num_groups = len(groups)

    # compute confusion tensor
    conftens = (
        confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)
        + 1e-32
    )
    # normalization constant
    norm = conftens.sum(axis=1)

    # initialize distance matrix
    dist_mat = np.zeros((num_groups, num_groups))

    # distance confusion matrix across groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :] / norm[k, :]
        for j in range(k + 1, num_groups):
            confmat_j = conftens[j, :, :] / norm[j, :]
            dist_mat[k, j] = np.abs((confmat_k - confmat_j).sum(axis=1)).sum() / (
                2 * num_classes
            )

    if aggregation_fun == "max":
        res = np.max(dist_mat)
    else:
        res = np.sum(dist_mat) / (num_groups * (num_groups - 1) / 2)

    return res


def multiclass_true_rates(
    p_attr, y_pred, y_true, groups=None, classes=None, aggregation_fun="mean"
):
    """
    Multiclass True Rates.

    Description
    ----------
    This metric is a multiclass generalisation of TPR
    Difference. For each group, compute the matrix of error
    rates (normalised confusion matrix). Compute all distances
    (mean absolute deviation) between the diagonals of such
    matrices. Then aggregate them using the mean, or max strategy.

    Interpretation
    --------------
    The accepted values and bounds for this metric are the same
    as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    y_true : array-like
        Target vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order

    Returns
    -------
    pandas DataFrame
        Accuracy Matrix : shape (num_groups, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import multiclass_true_rates
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> multiclass_true_rates(p_attr, y_pred, y_true)
    0.6666666666666666
    """
    # check and coerce inputs
    p_attr, y_pred, y_true, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=groups,
        classes=classes,
    )

    # variables
    num_groups = len(groups)

    # compute confusion tensor
    conftens = (
        confusion_tensor(p_attr, y_pred, y_true, groups, classes, as_tensor=True)
        + 1e-32
    )
    # normalization constant
    norm = conftens.sum(axis=1)

    # initialize distance matrix
    dist_mat = np.zeros((num_groups, num_groups))

    # distance confusion matrix across groups
    for k in range(num_groups):
        confmat_k = conftens[k, :, :] / norm[k, :]
        for j in range(k + 1, num_groups):
            confmat_j = conftens[j, :, :] / norm[j, :]
            dist_mat[k, j] = np.abs(np.diag(confmat_k - confmat_j)).mean()

    if aggregation_fun == "max":
        res = np.max(dist_mat)
    else:
        res = np.sum(dist_mat) / (num_groups * (num_groups - 1) / 2)

    return res


def multiclass_statistical_parity(
    p_attr, y_pred, groups=None, classes=None, aggregation_fun="mean"
):
    """
    Multiclass statistical parity.

    Description
    ----------
    This function computes statistical parity for a classification task
    with multiple classes and a protected attribute with multiple groups.
    For each group compute the vector of success rates for entering
    each class. Compute all distances (mean absolute deviation) between
    such vectors. Then aggregate them using the mean, or max strategy.

    Interpretation
    --------------
    The accepted values and bounds for this metric are the same
    as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

    Parameters
    ----------
    p_attr : array-like
        Multiclass protected attribute vector.
    y_pred : array-like
        Prediction vector (categorical).
    groups (optional) : list
        Unique groups from p_attr in order
    classes (optional) : list
        The unique output classes in order
    aggregation_fun (optional) : str
        The function to aggregate across groups ('mean' or 'max')

    Returns
    -------
    float
        Multiclass Statistical Parity

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import multiclass_statistical_parity
    >>> p_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'])
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> multiclass_statistical_parity(p_attr, y_pred, aggregation_fun='max')
    0.5
    """
    # check and coerce inputs
    p_attr, y_pred, _, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=groups,
        classes=classes,
    )

    # variables
    num_groups = len(groups)

    # compute frequency matrix (normalised by class)
    sr_mat = frequency_matrix(p_attr, y_pred, groups, classes).to_numpy() + 1e-32

    # initialize distance matrix
    dist_mat = np.zeros((num_groups, num_groups))

    # distance confusion matrix across groups
    for k in range(num_groups):
        pred_prob_k = sr_mat[k]
        for j in range(k + 1, num_groups):
            pred_prob_j = sr_mat[j]
            dist_mat[k, j] = np.abs(pred_prob_k - pred_prob_j).sum() / 2

    if aggregation_fun == "max":
        res = np.max(dist_mat)
    else:
        res = np.sum(dist_mat) / (num_groups * (num_groups - 1) / 2)

    return res


def multiclass_bias_metrics(
    p_attr, y_pred, y_true, groups=None, classes=None, metric_type="equal_outcome"
):
    """
    Multiclass bias metrics batch computation.

    Description
    ----------
    This function computes all the relevant multiclass bias metrics,
    and displays them as a pandas dataframe.

    Parameters
    ----------
    g_min : numpy array
        Minority class vector
    g_maj : numpy array
        Majority class vector
    y_pred : array-like
        Regression predictions vector
    y_true (optional) : numpy array
        Regression target vector
    metric_type : 'both', 'equal_outcome' or 'equal_opportunity'
        Specifies which metrics we compute

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """

    perform = {
        "Max Multiclass Statistical Parity": multiclass_statistical_parity,
        "Mean Multiclass Statistical Parity": multiclass_statistical_parity,
    }

    true_perform = {
        "Max Multiclass Equality of Opportunity": multiclass_equality_of_opp,
        "Max Multiclass Average Odds": multiclass_average_odds,
        "Max Multiclass True Positive Difference": multiclass_true_rates,
        "Mean Multiclass Equality of Opportunity": multiclass_equality_of_opp,
        "Mean Multiclass Average Odds": multiclass_average_odds,
        "Mean Multiclass True Positive Difference": multiclass_true_rates,
    }

    ref_vals = {
        "Max Multiclass Statistical Parity": 0,
        "Mean Multiclass Statistical Parity": 0,
        "Max Multiclass Equality of Opportunity": 0,
        "Max Multiclass Average Odds": 0,
        "Max Multiclass True Positive Difference": 0,
        "Mean Multiclass Equality of Opportunity": 0,
        "Mean Multiclass Average Odds": 0,
        "Mean Multiclass True Positive Difference": 0,
    }

    param = {
        "Max Multiclass Statistical Parity": "max",
        "Mean Multiclass Statistical Parity": "mean",
        "Max Multiclass Equality of Opportunity": "max",
        "Max Multiclass Average Odds": "max",
        "Max Multiclass True Positive Difference": "max",
        "Mean Multiclass Equality of Opportunity": "mean",
        "Mean Multiclass Average Odds": "mean",
        "Mean Multiclass True Positive Difference": "mean",
        "Max Multiclass Statistical Paraity": "max",
        "Mean Multiclass Statistical Paraity": "mean",
    }

    out_metrics = []
    opp_metrics = []

    out_metrics += [
        [
            pf,
            fn(p_attr, y_pred, groups, classes, aggregation_fun=param[pf]),
            ref_vals[pf],
        ]
        for pf, fn in perform.items()
    ]
    if y_true is not None:
        opp_metrics += [
            [
                pf,
                fn(p_attr, y_pred, y_true, groups, classes, aggregation_fun=param[pf]),
                ref_vals[pf],
            ]
            for pf, fn in true_perform.items()
        ]

    if metric_type == "both":
        metrics = out_metrics + opp_metrics
        return pd.DataFrame(
            metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    elif metric_type == "equal_outcome":
        return pd.DataFrame(
            out_metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    elif metric_type == "equal_opportunity":
        return pd.DataFrame(
            opp_metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    else:
        raise ValueError(
            "metric_type is not one of : both, equal_outcome, equal_opportunity"
        )
