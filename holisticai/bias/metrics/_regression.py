# Base Imports
import numpy as np
import pandas as pd

# utils
from ...utils._formatting import slice_arrays_by_quantile
from ...utils._validation import _check_non_empty, _regression_checks


def disparate_impact_regression(group_a, group_b, y_pred, q=0.8):
    r"""
    Disparate Impact quantile (Regression version).

    Description
    -----------
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
    q (optional) : float or array, default=0.8
        quantile of predictions considered

    Returns
    -------
    float
        Disparate Impact (top %) : :math:`\frac{sr_a}{sr_b}`

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
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

    # numbers
    n_a = group_a.sum()
    n_b = group_b.sum()

    _, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, group_a, group_b]
    )

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
    """
    Statistical Parity quantile (Regression version).

    Description
    -----------
    This function computes the difference of success rates between group_a and
    group_b, where sucess means that the predicted score exceeds a given quantile.

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0 are unfair towards group_a.
    Values above 0 are unfair towards group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q (optional) : float or array, default=0.5
        quantile of predictions considered

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
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

    # numbers
    n_a = group_a.sum()
    n_b = group_b.sum()

    _, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, group_a, group_b]
    )

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
    """
    No disparate impact level.

    Description
    -----------
    This function computes the maximum score such that thresholding at that score
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
    group_a, group_b, y_pred, _, _ = _regression_checks(
        group_a, group_b, y_pred, None, None
    )

    # grid
    q = np.linspace(1.0, 0.0, 100)

    # try different values
    for v in q:
        pred = np.quantile(y_pred, v)
        pass_members = y_pred >= pred
        a = sum(group_a * pass_members) / group_a.sum()
        b = sum(group_b * pass_members) / group_b.sum()
        # find score that does not allow adverse impact
        if b > 0:
            if 0.8 < a / b < 1.2:
                break
    return pred


def avg_score_diff(group_a, group_b, y_pred, q=0):
    """

    Average Score Difference.

    Description
    -----------
    This function computes the difference in average scores between
    group_a and group_b.

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. Negative values indicate the group_a
    has lower average score, so bias against group_a. Positive values
    indicate group_b has lower average score, so bias against group_b.
    Scale is relative to task.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        Average Score Spread : :math:`\texttt{AVgroup_a - AVgroup_b}`

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
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, group_a, group_b]
    )

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
    """

    Average Score Ratio.

    Description
    -----------
    This function computes the ratio in average scores between
    group_a and group_b.

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 1 is desired. Values below 1 indicate the group_a
    has lower average score, so bias against group_a. Values above 1
    indicate group_b has lower average score, so bias against group_b.
    (0.8, 1.25) range is considered fair.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        Average Score Ratio : :math:`\texttt{AVgroup_a / AVgroup_b}`

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
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, group_a, group_b]
    )

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
    """

    ZScore Difference.

    Description
    -----------
    This function computes the spread in Zscores between
    group_a and group_b. The Zscore is a normalised
    version of Disparate Impact.

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. The Zscore will approximate the number
    of standard deviations away from the mean. In particular values that
    exceed 2 are statistically significant with 95% probability.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        ZScore Difference : :math:`\frac{\texttt{AVgroup_a} - \texttt{AVgroup_b}}{\texttt{STD_pool}}`

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
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

    y_pred_list, group_a_list, group_b_list = slice_arrays_by_quantile(
        q, y_pred, [y_pred, group_a, group_b]
    )

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
            (
                (n_a - 1) * y_pred[group_a == 1].std() ** 2.0
                + (n_b - 1) * y_pred[group_b == 1].std() ** 2.0
            )
            / (n_b + n_a - 2)
        )

        zscore_diff[i] = (av_group_a - av_group_b) / std_pool

    return np.squeeze(zscore_diff)[()]


def statistical_parity_auc(group_a, group_b, y_pred):
    """
    Statistical parity (AUC).

    Description
    -----------
    This function computes the area under the statistical parity
    versus threshold curve.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.075 are considered
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
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)**3))
    >>> statistical_parity_auc(group_a, group_b, y_pred)
    0.12106666666666668
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(
        group_a, group_b, y_pred, None, None
    )

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
    """
    Weighed Statistical parity (AUC).

    Description
    -----------
    This function computes the area under the statistical
    parity versus threshold curve, weighed by the 2t distribution.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.1 are considered
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
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)**3))
    >>> _weighed_statistical_parity_auc(group_a, group_b, y_pred)
    0.12106666666666666
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(
        group_a, group_b, y_pred, None, None
    )

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
    """
    Max absolute statistical parity.

    Description
    -----------
    This function computes the maximum over all thresholds of
    the absolute statistical parity between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. Values below 0.1 in absolute value are
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
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)**3))
    >>> max_statistical_parity(group_a, group_b, y_pred)
    0.20000000000000007
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(
        group_a, group_b, y_pred, None, None
    )

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
    """
    Correlation difference.

    Description
    -----------
    This function computes the difference in correlation between predictions
    and targets for group_a and group_b.

    If q is a vector, this function returns a vector with the
    respective result for each given quantile in q.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -2 and 2,
    with -1 indicating strong bias against group_a, and +1
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
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        correlation difference : :math:`CV_a - CV_b`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import correlation_diff
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)**3))
    >>> y_true = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)**2))
    >>> correlation_diff(group_a, group_b, y_pred, y_true, q=0)
    1.0000000000000002
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, q = _regression_checks(
        group_a, group_b, y_pred, None, q
    )

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
    """
    RMSE ratio.

    Description
    -----------
    This function computes the ratio of the RMSE for group_a and group_b.

    If q is a vector, this function returns a vector with the
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
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        RMSE ratio : :math:`\frac{RMSE_a}{RMSE_b}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import rmse_ratio
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)))
    >>> y_true = np.concatenate((np.linspace(-1, 1, 50)**2, np.linspace(-1, 1, 50)**3))
    >>> rmse_ratio(group_a, group_b, y_pred, y_true)
    2.7471209467641367
    """
    # check and coerce inputs
    group_a, group_b, y_pred, y_true, q = _regression_checks(
        group_a, group_b, y_pred, y_true, q
    )

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
    """
    MAE ratio

    Description
    -----------
    This function computes the ratio of the MAE for group_a and group_b.

    If q is a vector, this function returns a vector with the
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
    q (optional) : float or array
        quantile of predictions considered

    Returns
    -------
    float
        MAE ratio : :math:`\frac{MAE_a}{MAE_b}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import mae_ratio
    >>> group_a = np.array([1] * 50 + [0] * 50)
    >>> group_b = np.array([0] * 50 + [1] * 50)
    >>> y_pred = np.concatenate((np.linspace(-1, 1, 50), np.linspace(-1, 1, 50)))
    >>> y_true = np.concatenate((np.linspace(-1, 1, 50)**2, np.linspace(-1, 1, 50)**3))
    >>> mae_ratio(group_a, group_b, y_pred, y_true)
    2.084201388888889
    """
    # check and coerce inputs
    group_a, group_b, y_pred, y_true, q = _regression_checks(
        group_a, group_b, y_pred, y_true, q
    )

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


def regression_bias_metrics(
    group_a, group_b, y_pred, y_true=None, metric_type="equal_outcome"
):
    """
    Regression bias metrics batch computation

    Description
    -----------
    This function computes all the relevant regression bias metrics,
    and displays them as a pandas dataframe.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    y_true (optional) : numpy array
        Target vector (regression)
    metric_type : 'both', 'equal_outcome' or 'equal_opportunity'
        Specifies which metrics we compute

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """

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

    out_metrics = [
        [pf, fn(group_a, group_b, y_pred, **hypers[pf]), ref_vals[pf]]
        for pf, fn in equal_outcome_metrics.items()
    ]
    if y_true is not None:
        opp_metrics = [
            [pf, fn(group_a, group_b, y_pred, y_true, **hypers[pf]), ref_vals[pf]]
            for pf, fn in equal_opportunity_metrics.items()
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
