# Base Imports
import numpy as np
import pandas as pd

# Efficacy metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# utils
from ...utils._validation import (
    _check_binary,
    _classification_checks,
    _regression_checks,
)


def _group_success_rate(g, y):
    """
    Group success rate.

    Description
    -----------
    This function computes the success rate for a given subgroup.

    Parameters
    ----------
    g : array-like
        subgroup vector (binary)
    y : array-like
        predictions vector (binary)

    Returns
    -------
    float
        group success rate

    """
    success_rate = y[g == 1].sum() / g.sum()  # success rate group_a
    return success_rate


def statistical_parity(group_a, group_b, y_pred):
    """
    Statistical parity.

    Description
    -----------
    This function computes the statistical parity (difference of success rates)
    between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. Negative values are unfair towards group_a.
    Positive values are unfair towards group_b. The range (-0.1,0.1)
    is considered acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)

    Returns
    -------
    float
        Statistical Parity : :math:`sr_a - sr_b`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import statistical_parity
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    >>> statistical_parity(group_a, group_b, y_pred)
    0.4166666666666667
    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b

    return sr_a - sr_b


def disparate_impact(group_a, group_b, y_pred):
    """
    Disparate Impact.

    Description
    -----------
    This function computes the disparate impact (ratio of success rates)
    between group_a and group_b class.

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
        Predictions vector (binary)

    Returns
    -------
    float
        Disparate Impact : :math:`sr_a/sr_b`

    References
    ----------
    .. [1] `M B Zafar, I Valera, M G Rodriguez, K P. Gummadi (2017).
            Fairness Constraints: Mechanisms for Fair Classification, MPI-SWS
            <https://arxiv.org/pdf/1507.05259.pdf>`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import disparate_impact
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    >>> disparate_impact(group_a, group_b, y_pred)
    2.25
    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b

    return sr_a / sr_b


def four_fifths(group_a, group_b, y_pred):
    """
    Four Fifths.

    Description
    -----------
    This function computes the four fifths rule (ratio of success rates)
    between group_a and group_b. We return the minimum of the ratio
    taken both ways.

    Interpretation
    --------------
    A value of 1 is desired. Values below 1 are unfair. The range (0.8,1)
    is considered acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)

    Returns
    -------
    float
        Four Fifths

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import four_fifths
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    >>> four_fifths(group_a, group_b, y_pred)
    0.4444444444444444
    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b

    return min(sr_a / sr_b, sr_b / sr_a)


def cohen_d(group_a, group_b, y_pred):
    r"""
    Cohen D.

    Description
    -----------
    This function computes the Cohen D statistic (normalised statistical parity)
    between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. Negative values are unfair towards group_a.
    Positive values are unfair towards group_b. Reference values:
    0.2 is considered a small effect size, 0.5 is considered medium,
    0.8 is considered large.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)

    Returns
    -------
    float
        Cohen D : :math:`\frac{sr_a-sr_b}{\texttt{std_pool}}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import cohen_d
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> cohen_d(group_a, group_b, y_pred)
    -0.7844645405527363
    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b

    # calculate STD_a and STD_b
    std_b = np.sqrt(sr_b * (1 - sr_b))
    std_a = np.sqrt(sr_a * (1 - sr_a))

    # calculate n_a and n_b
    n_a = group_a.sum()
    n_b = group_b.sum()

    # calculate poolSTD
    std_pool = np.sqrt(
        ((n_b - 1) * std_b**2 + (n_a - 1) * std_a**2) / (n_a + n_b - 2)
    )

    return (sr_a - sr_b) / std_pool


def z_test_diff(group_a, group_b, y_pred):
    r"""
    Z Test (Difference).

    Description
    -----------
    This function computes the Z-test statistic for the difference
    in success rates. Also known as 2-SD Statistic.

    Interpretation
    --------------
    A value of 0 is desired. This test considers the data unfair
    if the computed value is greater than 2 or smaller than -2,
    indicating a statistically significant difference in success rates.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)

    Returns
    -------
    float
        Z test (difference version)

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import z_test_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    >>> z_test_diff(group_a, group_b, y_pred)
    1.290994449

    References
    ----------
    .. [1] `Morris (2001).
           Sample size requirements for adverse impact analysis
            <https://www.semanticscholar.org/paper/Sample-Size-Required-for-Adverse-Impact-Analysis-Morris/877f7acd7c646a21f4947166a07f41664dcabe95>`

    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b
    n_a = group_a.sum()
    n_b = group_b.sum()
    sr_tot = (sr_a * n_a + sr_b * n_b) / (n_a + n_b)
    n_tot = n_a + n_b

    # calculate p_a
    p_a = n_a / n_tot

    return (sr_a - sr_b) / np.sqrt((sr_tot * (1 - sr_tot)) / (n_tot * p_a * (1 - p_a)))


def z_test_ratio(group_a, group_b, y_pred):
    r"""
    Z Test (Ratio).

    Description
    -----------
    This function computes the Z-test statistic for the ratio
    in success rates. Also known as 2-SD Statistic.

    Interpretation
    --------------
    A value of 0 is desired. This test considers the data unfair
    if the computed value is greater than 2 or smaller than -2,
    indicating a statistically significant ratio in success rates.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)

    Returns
    -------
    float
        Z-test (ratio version)

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import z_test_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    >>> z_test_ratio(group_a, group_b, y_pred)
    1.256287689

    References
    ----------
    .. [1] `Morris (2001).
           Sample size requirements for adverse impact analysis
            <https://www.semanticscholar.org/paper/Sample-Size-Required-for-Adverse-Impact-Analysis-Morris/877f7acd7c646a21f4947166a07f41664dcabe95>`
    """
    # check and coerce
    group_a, group_b, y_pred, _ = _classification_checks(
        group_a, group_b, y_pred, y_true=None
    )

    # calculate sr_a and sr_b
    sr_a = _group_success_rate(group_a, y_pred)  # success rate group_a
    sr_b = _group_success_rate(group_b, y_pred)  # success rate group_b
    n_a = group_a.sum()
    n_b = group_b.sum()
    sr_tot = (sr_a * n_a + sr_b * n_b) / (n_a + n_b)
    n_tot = n_a + n_b

    # calculate p_a
    p_a = n_a / n_tot

    return (np.log(sr_a / sr_b)) / np.sqrt(
        (1 - sr_tot) / (sr_tot * n_tot * p_a * (1 - p_a))
    )


def _correlation_diff(group_a, group_b, y_pred, y_true):
    """
    Correlation difference.

    Description
    -----------
    This function computes the difference in correlation between predicted
    and true labels for group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -2 and 2,
    with negative values indicating bias against group_a, and
    positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        Correlation Difference : :math:`CV_a - CV_b`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import correlation_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> correlation_diff(group_a, group_b, y_pred, y_true)
    1.4472135954999579
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Calculate Pearson correlations
    cv_a = np.corrcoef(y_pred[group_a == 1], y_true[group_a == 1])[1, 0]
    cv_b = np.corrcoef(y_pred[group_b == 1], y_true[group_b == 1])[1, 0]

    return cv_a - cv_b


def equal_opportunity_diff(group_a, group_b, y_pred, y_true):
    """
    Equality of opportunity difference.

    Description
    -----------
    This function computes the difference in true positive
    rates for group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a, and
    positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        Equal opportunity difference : :math:`tpr_a - tpr_b`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import equal_opportunity_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> equal_opportunity_diff(group_a, group_b, y_pred, y_true)
    0.33333333333333337
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Calculate true positive rates
    tpr_a = confusion_matrix(
        y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
    )[1, 1]
    tpr_b = confusion_matrix(
        y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
    )[1, 1]

    return tpr_a - tpr_b


def false_positive_rate_diff(group_a, group_b, y_pred, y_true):
    """
    False positive rate difference.

    Description
    ----------
    This function computes the difference in false positive
    rates between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a, and
    positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        FPR_diff : :math:`fpr_a - fpr_b`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import false_positive_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> false_positive_diff(group_a, group_b, y_pred, y_true)
    -1.0
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Calculate false positive rates
    fpr_a = confusion_matrix(
        y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
    )[0, 1]
    fpr_b = confusion_matrix(
        y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
    )[0, 1]

    return fpr_a - fpr_b


def false_negative_rate_diff(group_a, group_b, y_pred, y_true):
    """
    False negative Rate difference.

    Description
    ----------
    This function computes the difference in false negative
    rates for group_a and group_b.

    Interpretation
    ----------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_b, and
    positive values indicating bias against group_a.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        False Negative Rate difference : :math:`fnr_a - fnr_b`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import fnr_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> fnr_diff(group_a, group_b, y_pred, y_true)
    -0.3333333333333333
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Calculate false negative rates
    fnr_a = confusion_matrix(
        y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
    )[1, 0]
    fnr_b = confusion_matrix(
        y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
    )[1, 0]

    return fnr_a - fnr_b


def true_negative_rate_diff(group_a, group_b, y_pred, y_true):
    """
    True negative Rate difference.

    Description
    ----------
    This function computes the difference in true negative
    rates for group_a and group_b.

    Interpretation
    ----------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a, and
    positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        True Negative Rate difference : :math:`tnr_a - tnr_b`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import tnr_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> tnr_diff(group_a, group_b, y_pred, y_true)
    1
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Calculate false negative rates
    tnr_a = confusion_matrix(
        y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
    )[0, 0]
    tnr_b = confusion_matrix(
        y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
    )[0, 0]

    return tnr_a - tnr_b


def average_odds_diff(group_a, group_b, y_pred, y_true):
    """
    Average Odds Difference.

    Description
    -----------
    This function computes the difference in average odds
    between group_a and group_b.

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a,
    and positive values indicating bias against group_b.
    The range (-0.1,0.1) is considered acceptable.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        AOD : :math:`0.5 * (fpr_a-fpr_b + tpr_a-tpr_b)`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import average_odds_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> average_odds_diff(group_a, group_b, y_pred, y_true)
    -0.3333333333333333
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(
        group_a, group_b, y_pred, y_true
    )

    # Compute AOD
    aod = 0.5 * (
        equal_opportunity_diff(group_a, group_b, y_pred, y_true)
        + false_positive_rate_diff(group_a, group_b, y_pred, y_true)
    )

    return aod


def accuracy_diff(group_a, group_b, y_score, y_true):
    """
    Accuracy Difference.

    Description
    ----------
    This function computes the difference in accuracy
    of predictions for group_a and group_b

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a,
    and positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_score : numpy array
        Probability estimates (regression)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        acc_diff : acc_a - acc_b

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import accuracy_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    >>> accuracy_diff(group_a, group_b, y_pred, y_true)
    0.6666666666666667
    """
    # check and coerce
    group_a, group_b, y_score, y_true, _ = _regression_checks(
        group_a, group_b, y_score, y_true, None
    )
    _check_binary(y_true, "y_true")

    # split data by groups
    y_true_a = y_true[group_a == 1]
    y_score_a = y_score[group_a == 1]
    y_true_b = y_true[group_b == 1]
    y_score_b = y_score[group_b == 1]

    # compute abroca
    val = accuracy_score(y_true_a, y_score_a) - accuracy_score(y_true_b, y_score_b)
    return val


def abroca(group_a, group_b, y_score, y_true):
    """
    ABROCA (area between roc curves).

    Description
    -----------
    This function computes the area between the roc curve
    of group_a and the roc curve of group_b

    Interpretation
    --------------
    A value of 0 is desired. This metric ranges between -1 and 1,
    with negative values indicating bias against group_a,
    and positive values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_score : array-like
        Probability estimates (regression)
    y_true : array-like
        Target vector (binary)

    Returns
    -------
    float
        ABROCA : roc_auc_a - roc_auc_b

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import abroca
    >>> group_a = np.array([1]*50 + [0]*50)
    >>> group_b = np.array([0]*50 + [1]*50)
    >>> y_score = np.concatenate((np.linspace(0,1,50),np.linspace(0,1,50)**2))
    >>> y_true = y_score + np.random.random(y_score.shape)>0.5
    >>> abroca(group_a, group_b, y_score, y_true)
    0.11806478405315601
    """
    # check and coerce
    group_a, group_b, y_score, y_true, _ = _regression_checks(
        group_a, group_b, y_score, y_true, None
    )
    _check_binary(y_true, "y_true")

    # split data by groups
    y_true_a = y_true[group_a == 1]
    y_score_a = y_score[group_a == 1]
    y_true_b = y_true[group_b == 1]
    y_score_b = y_score[group_b == 1]

    # compute abroca
    val = roc_auc_score(y_true_a, y_score_a) - roc_auc_score(y_true_b, y_score_b)
    return val


def classification_bias_metrics(
    group_a, group_b, y_pred, y_true=None, y_score=None, metric_type="equal_outcome"
):
    """
    Classification bias metrics batch computation.

    Description
    -----------
    This function computes all the relevant classification bias metrics,
    and displays them as a pandas dataframe. It also includes a fair reference
    value for comparison.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (binary)
    y_true (optional) : numpy array
        Target vector (binary)
    y_score (optional) : array-like
        Probability estimates (regression)
    metric_type (optional) : 'both', 'equal_outcome' or 'equal_opportunity'
        Specifies which metrics we compute

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference

    Examples
    --------
    """
    equal_outcome_metrics = {
        "Statistical Parity": statistical_parity,
        "Disparate Impact": disparate_impact,
        "Four Fifths Rule": four_fifths,
        "Cohen D": cohen_d,
        "2SD Rule": z_test_diff,
    }

    equal_opportunity_metrics = {
        "Equality of Opportunity Difference": equal_opportunity_diff,
        "False Positive Rate Difference": false_positive_rate_diff,
        "Average Odds Difference": average_odds_diff,
        "Accuracy Difference": accuracy_diff,
    }

    soft_metrics = {
        "ABROCA": abroca,
    }

    ref_vals = {
        "Statistical Parity": 0,
        "Disparate Impact": 1,
        "Four Fifths Rule": 1,
        "Cohen D": 0,
        "Equality of Opportunity Difference": 0,
        "False Positive Rate Difference": 0,
        "Average Odds Difference": 0,
        "Accuracy Difference": 0,
        "ABROCA": 0,
        "2SD Rule": 0,
    }

    out_metrics = [
        [pf, fn(group_a, group_b, y_pred), ref_vals[pf]]
        for pf, fn in equal_outcome_metrics.items()
    ]
    if y_true is not None:
        opp_metrics = [
            [pf, fn(group_a, group_b, y_pred, y_true), ref_vals[pf]]
            for pf, fn in equal_opportunity_metrics.items()
        ]
    if y_score is not None:
        opp_metrics += [
            [pf, fn(group_a, group_b, y_score, y_true), ref_vals[pf]]
            for pf, fn in soft_metrics.items()
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
