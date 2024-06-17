# Base Imports
import warnings

import numpy as np
import pandas as pd

# utils
from holisticai.utils._formatting import slice_arrays_by_quantile
from holisticai.utils._validation import _check_non_empty, _regression_checks


def _calc_success_rate(group_membership: np.array, threshold=float):
    return (group_membership > threshold).mean()


def success_rate_regression(group_a, group_b, y_pred, threshold=0.50):
    """Success rate (Regression version)

    Calculates the raw success rates for each group.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (numerical)
    threshold: float, str, optional
        The number above which the result is considered a success. Ranged between 0 and 1.
        Also accepts 'median' and 'mean'.

    Returns
    -------
    dict
        Dictionary with two keys, sr_a and sr_b (success rate for group a and b)
    """
    # Needs to be numpy array or the following operations won't be correct
    if isinstance(threshold, str) and (threshold not in {"median", "mean"}):
        msg = "Threshold not recognised"
        raise ValueError(msg)
    if threshold == "median":
        threshold = np.median(y_pred)
    if threshold == "mean":
        threshold = np.mean(y_pred)
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    y_pred = np.array(y_pred)
    group_a_membership = y_pred[group_a == 1]
    group_b_membership = y_pred[group_b == 1]
    sr_a = _calc_success_rate(group_a_membership, threshold)
    sr_b = _calc_success_rate(group_b_membership, threshold)  # success rate group_b
    return {"sr_a": sr_a, "sr_b": sr_b}


def disparate_impact_regression(group_a, group_b, y_pred, q=0.8):
    r"""Disparate Impact quantile (Regression version)

    This function computes the ratio of success rates between group_a and
    group_b, where sucess means predicted score exceeds a given quantile (default = 0.8).

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 1 is desired. Values below 1 are unfair towards group_a.
    Values above 1 are unfair towards group_b. The range (0.8,1.2)
    is considered acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.8

    Returns
    -------
    float
        Disparate Impact (top %)

    Notes
    -----
    :math:`\frac{sr_a}{sr_b}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import disparate_impact_regression
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.5])
    >>> disparate_impact_regression(group_a, group_b, y_pred, q=0.7)
    1.5
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    # numbers
    n_a = group_a.sum()
    n_b = group_b.sum()

    _, group_a_list, group_b_list = slice_arrays_by_quantile(q, y_pred, [y_pred, group_a, group_b])

    n_q = len(group_a_list)
    disp_impact = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        sr_a = group_a.sum() / n_a  # success rate group_a
        sr_b = group_b.sum() / n_b  # success rate group_b

        disp_impact[i] = sr_a / sr_b

    return np.squeeze(disp_impact)[()]


def statistical_parity_regression(group_a, group_b, y_pred, q=0.5):
    """Statistical Parity quantile (Regression version)

    This function computes the difference of success rates between group_a and\
    group_b, where sucess means that the predicted score exceeds a given quantile.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0 are unfair towards group_a.\
    Values above 0 are unfair towards group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.5

    Returns
    -------
    float
        Statistical Parity (top %) : SR_a - SR_b

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import statistical_parity_regression
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.5])
    >>> statistical_parity_regression(group_a, group_b, y_pred, q=0.7)
    0.16666666666666669
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    # numbers
    n_a = group_a.sum()
    n_b = group_b.sum()

    _, group_a_list, group_b_list = slice_arrays_by_quantile(q, y_pred, [y_pred, group_a, group_b])

    n_q = len(group_a_list)
    stat_parity = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        sr_a = group_a.sum() / n_a  # success rate group_a
        sr_b = group_b.sum() / n_b  # success rate group_b

        stat_parity[i] = sr_a - sr_b

    return np.squeeze(stat_parity)[()]


def no_disparate_impact_level(group_a, group_b, y_pred):
    """No disparate impact level

    This function computes the maximum score such that thresholding at that score\
    does not allow adverse impact.

    Interpretation
    --------------

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)

    Returns
    -------
    float
        No disparate impact level

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import no_disparate_impact_level
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.5])
    >>> no_disparate_impact_level(group_a, group_b, y_pred)
    0.7
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # grid
    q = np.linspace(1.0, 0.0, 100)

    # try different values
    for v in q:
        pred = np.quantile(y_pred, v)
        pass_members = y_pred >= pred
        a = sum(group_a * pass_members) / group_a.sum()
        b = sum(group_b * pass_members) / group_b.sum()
        # find score that does not allow adverse impact
        lower_bound = 0.8
        upper_bound = 1.2
        if b > 0 and lower_bound < a / b < upper_bound:
            break
    return pred


def avg_score_diff(group_a, group_b, y_pred, q=0):
    """Average Score Difference

    This function computes the difference in average scores between\
    group_a and group_b.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. Negative values indicate the group_a\
    has lower average score, so bias against group_a. Positive values\
    indicate group_b has lower average score, so bias against group_b.\
    Scale is relative to task.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        Average Score Spread

    Notes
    -----
    :math:`\texttt{AVgroup_a - AVgroup_b}`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_score_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.5])
    >>> avg_score_diff(group_a, group_b, y_pred)
    -0.13333333333333341
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(q, y_pred, [y_pred, group_a, group_b])

    n_q = len(group_a_list)
    avg_score_diff = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        y_pred = y_pred_list[i]
        avgroup_a = y_pred[group_a == 1].mean()
        avgroup_b = y_pred[group_b == 1].mean()

        avg_score_diff[i] = avgroup_a - avgroup_b

    return np.squeeze(avg_score_diff)[()]


def avg_score_ratio(group_a, group_b, y_pred, q=0):
    """Average Score Ratio

    This function computes the ratio in average scores between\
    group_a and group_b.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 1 is desired. Values below 1 indicate the group_a\
    has lower average score, so bias against group_a. Values above 1\
    indicate group_b has lower average score, so bias against group_b.\
    (0.8, 1.25) range is considered fair.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        Average Score Ratio

    Notes
    -----
    :math:`\texttt{AVgroup_a / AVgroup_b}`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_score_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.7, 0.9, 0.8, 0.6, 0.3, 0.5])
    >>> avg_score_ratio(group_a, group_b, y_pred)
    0.7894736842
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(q, y_pred, [y_pred, group_a, group_b])

    n_q = len(group_a_list)
    avg_score_ratio = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        y_pred = y_pred_list[i]
        avgroup_a = y_pred[group_a == 1].mean()
        avgroup_b = y_pred[group_b == 1].mean()

        avg_score_ratio[i] = avgroup_a / avgroup_b

    return np.squeeze(avg_score_ratio)[()]


def zscore_diff(group_a, group_b, y_pred, q=0):
    """ZScore Difference

    This function computes the spread in Zscores between\
    group_a and group_b. The Zscore is a normalised\
    version of Disparate Impact.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. The Zscore will approximate the number\
    of standard deviations away from the mean. In particular values that\
    exceed 2 are statistically significant with 95% probability.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        ZScore Difference

    Notes
    -----
    :math:`\frac{\texttt{AVgroup_a} - \texttt{AVgroup_b}}{\texttt{STD_pool}}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import zscore_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.8, 0.9, 0.2, 0.1, 0.2, 0.9, 0.3, 0.6, 0.3, 0.5])
    >>> zscore_diff(group_a, group_b, y_pred)
    0.1166919931983158
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(q, y_pred, [y_pred, group_a, group_b])

    n_q = len(group_a_list)
    zscore_diff = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        # Get means
        y_pred = y_pred_list[i]
        av_group_a = y_pred[group_a == 1].mean()
        av_group_b = y_pred[group_b == 1].mean()

        # Get n_a and n_b
        n_a = group_a.sum()
        n_b = group_b.sum()

        # STD pooled
        std_pool = np.sqrt(
            ((n_a - 1) * y_pred[group_a == 1].std() ** 2.0 + (n_b - 1) * y_pred[group_b == 1].std() ** 2.0)
            / (n_b + n_a - 2)
        )

        zscore_diff[i] = (av_group_a - av_group_b) / std_pool

    return np.squeeze(zscore_diff)[()]


def statistical_parity_auc(group_a, group_b, y_pred):
    """Statistical parity (AUC)

    This function computes the area under the statistical parity\
    versus threshold curve.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.075 are considered\
    acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)

    Returns
    -------
    float
        statistical parity (AUC)

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import statistical_parity_auc
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 3))
    >>> statistical_parity_auc(group_a, group_b, y_pred)
    0.12106666666666668
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # thresholds
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()
    di_arr = np.abs(pass_a - pass_b)

    # AUC
    return np.sum(di_arr * np.array([1 / 150] * 150))


def _weighed_statistical_parity_auc(group_a, group_b, y_pred):
    """Weighed Statistical parity (AUC)

    This function computes the area under the statistical\
    parity versus threshold curve, weighed by the 2t distribution.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.1 are considered\
    acceptable.

    Parameters
    ----------
    group_a : numpy array
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)

    Returns
    -------
    float
        Weighed statistical parity (AUC)

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import _weighed_statistical_parity_auc
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 3))
    >>> _weighed_statistical_parity_auc(group_a, group_b, y_pred)
    0.12106666666666666
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # thresholds
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()
    di_arr = np.abs(pass_a - pass_b)
    differentials = np.linspace(2, 0, 150)

    # Weighed AUC
    return np.sum(di_arr * differentials / 150)


def max_statistical_parity(group_a, group_b, y_pred):
    """Max absolute statistical parity

    This function computes the maximum over all thresholds of\
    the absolute statistical parity between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.1 in absolute value are\
    considered acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)

    Returns
    -------
    float
        max absolute statistical parity

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import max_statistical_parity
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 3))
    >>> max_statistical_parity(group_a, group_b, y_pred)
    0.20000000000000007
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # thresholds
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()
    di_arr = np.abs(pass_a - pass_b)

    # MAX
    return np.max(di_arr)


def correlation_diff(group_a, group_b, y_pred, y_true, q=0):
    """Correlation difference

    This function computes the difference in correlation between predictions\
    and targets for group_a and group_b.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -2 and 2,\
    with -1 indicating strong bias against group_a, and +1\
    indicating strong bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    y_true : numpy array
        Target vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        correlation difference

    Notes
    -----
    :math:`CV_a - CV_b`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import correlation_diff
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 3))
    >>> y_true = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 2))
    >>> correlation_diff(group_a, group_b, y_pred, y_true, q=0)
    1.0000000000000002
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(group_a, group_b, y_pred, None, q)

    y_pred_list, y_true_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, y_true, group_a, group_b]
    )

    n_q = len(group_a_list)
    corr_diff = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        # Compute Pearson correlations
        y_pred = y_pred_list[i]
        y_true = y_true_list[i]
        cv_a = np.corrcoef(y_pred[group_a == 1], y_true[group_a == 1])[1, 0]
        cv_b = np.corrcoef(y_pred[group_b == 1], y_true[group_b == 1])[1, 0]

        corr_diff[i] = cv_a - cv_b

    return np.squeeze(corr_diff)[()]


def rmse_ratio(group_a, group_b, y_pred, y_true, q=0):
    """RMSE ratio

    This function computes the ratio of the RMSE for group_a and group_b.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against group_a.
    Higher values show bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    y_true : numpy array
        Target vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        RMSE ratio

    Notes
    -----
    :math:`\frac{RMSE_a}{RMSE_b}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import rmse_ratio
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)))
    >>> y_true = np.concatenate(
    ...     (np.linspace(-1, 1, 50) ** 2, np.linspace(-1, 1, 50) ** 3)
    ... )
    >>> rmse_ratio(group_a, group_b, y_pred, y_true)
    2.7471209467641367
    """
    # check and coerce inputs
    group_a, group_b, y_pred, y_true, q = _regression_checks(group_a, group_b, y_pred, y_true, q)

    y_pred_list, y_true_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, y_true, group_a, group_b]
    )

    n_q = len(group_a_list)
    rmse_ratio = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        # Compute RMSE for both groups
        y_pred = y_pred_list[i]
        y_true = y_true_list[i]
        rmse_a = np.sqrt(((y_true[group_a == 1] - y_pred[group_a == 1]) ** 2.0).mean())
        rmse_b = np.sqrt(((y_true[group_b == 1] - y_pred[group_b == 1]) ** 2.0).mean())

        rmse_ratio[i] = rmse_a / rmse_b

    return np.squeeze(rmse_ratio)[()]


def mae_ratio(group_a, group_b, y_pred, y_true, q=0):
    """MAE ratio

    This function computes the ratio of the MAE for group_a and group_b.

    If q is a vector, this function returns a vector with the\
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against group_a.
    Higher values show bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    y_true : numpy array
        Target vector (regression)
    q : float, array-like, optional
        quantile of predictions considered, default=0.

    Returns
    -------
    float
        MAE ratio

    Notes
    -----
    :math:`\frac{MAE_a}{MAE_b}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import mae_ratio
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)))
    >>> y_true = np.concatenate(
    ...     (np.linspace(-1, 1, 50) ** 2, np.linspace(-1, 1, 50) ** 3)
    ... )
    >>> mae_ratio(group_a, group_b, y_pred, y_true)
    2.084201388888889
    """
    # check and coerce inputs
    group_a, group_b, y_pred, y_true, q = _regression_checks(group_a, group_b, y_pred, y_true, q)

    y_pred_list, y_true_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, y_true, group_a, group_b]
    )

    n_q = len(group_a_list)
    mae_ratio = np.zeros(n_q)

    for i in range(n_q):
        group_a, group_b = group_a_list[i], group_b_list[i]
        _check_non_empty(group_a, name="group_a", quantile=q[i])
        _check_non_empty(group_b, name="group_b", quantile=q[i])

        # Compute MAE for both groups
        y_pred = y_pred_list[i]
        y_true = y_true_list[i]
        mae_a = (np.abs(y_true[group_a == 1] - y_pred[group_a == 1])).mean()
        mae_b = (np.abs(y_true[group_b == 1] - y_pred[group_b == 1])).mean()

        mae_ratio[i] = mae_a / mae_b

    return np.squeeze(mae_ratio)[()]


def regression_bias_metrics(group_a, group_b, y_pred, y_true=None, metric_type="group"):
    """Regression bias metrics batch computation

    This function computes all the relevant regression bias metrics,\
    and displays them as a pandas dataframe.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    y_true : numpy array, optional
        Target vector (regression)
    metric_type : str, optional
        Specifies which metrics we compute : individual, group, equal_outcome, equal_opportunity, both
    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """

    individual_metrics = {
        "Jain Index": jain_index,
    }

    equal_outcome_metrics = {
        "Disparate Impact Q90": disparate_impact_regression,
        "Disparate Impact Q80": disparate_impact_regression,
        "Disparate Impact Q50": disparate_impact_regression,
        "Statistical Parity Q50": statistical_parity_regression,
        "No Disparate Impact Level": no_disparate_impact_level,
        "Average Score Difference": avg_score_diff,
        "Average Score Ratio": avg_score_ratio,
        "Z Score Difference": zscore_diff,
        "Max Statistical Parity": max_statistical_parity,
        "Statistical Parity AUC": statistical_parity_auc,
    }

    equal_opportunity_metrics = {
        "RMSE Ratio": rmse_ratio,
        "RMSE Ratio Q80": rmse_ratio,
        "MAE Ratio": mae_ratio,
        "MAE Ratio Q80": mae_ratio,
        "Correlation Difference": correlation_diff,
    }

    ref_vals = {
        "Disparate Impact Q90": 1,
        "Disparate Impact Q80": 1,
        "Disparate Impact Q50": 1,
        "Statistical Parity Q50": 0,
        "No Disparate Impact Level": "-",
        "Average Score Difference": 0,
        "Average Score Ratio": 1,
        "Z Score Difference": 0,
        "Max Statistical Parity": 0,
        "Statistical Parity AUC": 0,
        "RMSE Ratio": 1,
        "RMSE Ratio Q80": 1,
        "MAE Ratio": 1,
        "MAE Ratio Q80": 1,
        "Correlation Difference": 0,
        "Jain Index": 1,
    }

    hypers = {
        "Disparate Impact Q90": {"q": 0.9},
        "Disparate Impact Q80": {"q": 0.8},
        "Disparate Impact Q50": {"q": 0.5},
        "Statistical Parity Q50": {"q": 0.8},
        "No Disparate Impact Level": {},
        "Average Score Difference": {},
        "Average Score Ratio": {},
        "Z Score Difference": {},
        "Max Statistical Parity": {},
        "Statistical Parity AUC": {},
        "RMSE Ratio": {},
        "RMSE Ratio Q80": {"q": 0.8},
        "MAE Ratio": {},
        "MAE Ratio Q80": {"q": 0.8},
        "Correlation Difference": {},
    }

    y_pred = np.squeeze(y_pred)

    if y_true is not None:
        y_true = np.squeeze(y_true)

    has_group_parameters = all((p is not None) for p in [group_a, group_b, y_pred])

    if has_group_parameters:
        out_metrics = [
            [pf, fn(group_a, group_b, y_pred, **hypers[pf]), ref_vals[pf]] for pf, fn in equal_outcome_metrics.items()
        ]
        if y_true is not None:
            opp_metrics = [
                [
                    pf,
                    fn(group_a, group_b, y_pred, y_true, **hypers[pf]),
                    ref_vals[pf],
                ]
                for pf, fn in equal_opportunity_metrics.items()
            ]

    if metric_type == "individual":
        indv_metrics = []
        if y_pred is not None and y_true is not None:
            indv_metrics += [[pf, fn(y_pred, y_true), ref_vals[pf]] for pf, fn in individual_metrics.items()]
        else:
            # in case of missing y_pred or y_true
            msg = "y_pred and y_true must be provided for individual metrics"
            raise ValueError(msg)

    if metric_type in ["group", "both"]:
        if metric_type == "both":
            # TODO: remove both for next version
            warnings.warn(
                "`both` option will be depreciated in the next versions, use group", DeprecationWarning, stacklevel=2
            )

        metrics = out_metrics + opp_metrics
        return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    if metric_type == "equal_outcome":
        return pd.DataFrame(out_metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    if metric_type == "equal_opportunity":
        return pd.DataFrame(opp_metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    if metric_type == "individual":
        return pd.DataFrame(indv_metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    msg = "metric_type is not one of : both, equal_outcome, equal_opportunity"
    raise ValueError(msg)


#### Individual metrics


def jain_index(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """The Jain index (Fairness index)

    The Jain index is an index proposed for resources allocation that measures the "equality" of user allocation [1].
    For our purposes, from the point of view of fairness, it measures the equality of the error distributed in the\
    model outcomes. Empirically, we could say that a model with a Jain index of 1 is a model that distributes the error\
    equally among all the samples.

    Please, use this metric with caution, as it is not a metric that has been proposed for fairness in machine learning\
    models, but for resources allocation.

    Interpretation
    --------------
    From the point of view of fairness, it measures the equality of the error distributed among the samples. A fairer\
    model will have a Jain index closer to 1.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)
        The predicted target values.

    Returns
    -------
    float
        The Jain index of the input array.

    References
    ----------
    .. [1] Jain, R. (1984). A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared
    Computer Systems. Eastern Research Laboratory, Digital Equipment Corporation.

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import jain_index
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1, 2, 3, 4, 4])
    >>> jain_index(y_pred, y_true)
    0.2
    """
    error = np.abs(y_true - y_pred)
    jain = ((error.sum()) ** 2) / (len(error) * (error**2).sum())
    if np.isnan(jain):
        return 1.0
    return jain
