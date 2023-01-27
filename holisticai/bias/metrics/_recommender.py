# Base imports
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.metrics import mean_absolute_error

# utils
from ...utils import mat_to_binary, normalize_tensor

# Recommender Efficacy Metrics
from ...utils._recommender_tools import (
    avg_f1,
    avg_precision,
    avg_recall,
    entropy,
    recommender_mae,
    recommender_rmse,
)
from ...utils._validation import _recommender_checks


def aggregate_diversity(mat_pred, top=None, thresh=0.5, normalize=False):
    r"""
    Aggregate Diversity.

    Description
    ----------
    Given a matrix of scores, this function computes the recommended items for
    each user, selecting either the highest-scored items or those above an input
    threshold. It then returns the aggregate diversity: the proportion of recommended
    items out of all possible items.

    Interpretation
    --------------
    A value of 1 is desired. We wish for a high proportion of items
    to be shown to avoid the 'rich get richer effect'.

    Parameters
    ----------
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items recommended to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Aggregate Diversity : :math:`\frac{|Items\; shown|}{|Items|}`

    References
    ----------
    .. [1] `H Abdollahpouri and M Mansoury and R Burke and B Mobasher and E Malthouse (2021).
            User-centered Evaluation of Popularity Bias in Recommender Systems, ACM.
            <https://doi.org/10.1145%2F3450613.3456821>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import aggregate_diversity
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> aggregate_diversity(mat_pred, top=None, thresh=0.8, normalize=True)
    0.75
    """
    # input checks and coerce
    _, _, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=None,
        group_b=None,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize scores
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Make matrix binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # Count items by summing over users
    item_count = binary_mat_pred.sum(axis=0)

    # Proportion of all items shown
    agg_div = (item_count >= 1).sum() / len(item_count)

    return agg_div


def gini_index(mat_pred, top=None, thresh=0.5, normalize=False):
    """
    GINI index.

    Description
    ----------
    Measures the inequality across the frequency distribution
    of the recommended items.

    Interpretation
    --------------
    An algorithm that recommends each item the same number of
    times (uniform distribution) will have a Gini index of 0
    and the one with extreme inequality will have a Gini of 1.

    Parameters
    ----------
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        GINI

    References
    ----------
    .. [1] `M Mansoury, H Abdollahpouri, M Pechenizkiy, B Mobasher, R Burke (2020).
            FairMatch: A Graph-based Approach for Improving Aggregate Diversity in Recommender Systems,
            <https://doi.org/10.48550/arXiv.2005.01148>`
    .. [2] `Farzad Eskandanian, Bamshad Mobasher (2020).
            Using Stable Matching to Optimize the Balance between Accuracy and Diversity in Recommendation
            <https://doi.org/10.48550/arXiv.2006.03715>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import gini_index
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> gini_index(mat_pred, top=2, thresh=None, normalize=False)
    0.1333333333333333
    """
    # input check and coerce
    _, _, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=None,
        group_b=None,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize score matrix
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Make matrix binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # compute frequencies and sort them
    item_nums = binary_mat_pred.sum(axis=0)
    item_freqs = item_nums / item_nums.sum()
    item_freqs_s = np.sort(item_freqs)

    # compute gini sum
    num_items = mat_pred.shape[1]
    w = 2 * np.arange(num_items) - num_items + 1
    gini_val = (w * item_freqs_s).sum() / (num_items - 1)

    return gini_val


def exposure_entropy(mat_pred, top=None, thresh=0.5, normalize=False):
    r"""
    Exposure Entropy.

    Description
    ----------
    This function measures the entropy of the item exposure distribution.

    Interpretation
    --------------
    A low entropy (close to 0) indicates high certainty as to which item
    will be shown. Higher entropies therefore ensure a more
    homogeneous distribution. Scale is relative to number of items.

    Parameters
    ----------
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Exposure Entropy : :math:`-\sum_{k}{ p_k} \ln(p_k)`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import exposure_entropy
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> exposure_entropy(mat_pred, top=None, thresh=0.3, normalize=True)
    1.3762266043445464
    """
    # input checks and coerce
    _, _, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=None,
        group_b=None,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize score matrix
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Get the item exposures
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)
    item_exposures = binary_mat_pred.sum(axis=0)
    item_exposure_dist = item_exposures / item_exposures.sum()

    # Return entropy
    return entropy(item_exposure_dist)


def avg_recommendation_popularity(mat_pred, top=None, thresh=0.5, normalize=False):
    """
    Average Recommendation Popularity.

    Description
    ----------
    This function computes the average recommendation popularity
    of items over users. We define the recommendation popularity
    as the average amount of times an item is recommended.

    Interpretation
    --------------
    A low value is desidered and suggests that items have been
    recommended equally across the population.

    Parameters
    ----------
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Average Recommendation Popularity

    References
    ----------
    .. [1] `M Mansoury, H Abdollahpouri, M Pechenizkiy, B Mobasher, R Burke, E Malthouse (2020).
            User-centered Evaluation of Popularity Bias in Recommender Systems,
            <https://arxiv.org/pdf/2103.06364.pdf>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_recommendation_popularity
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> avg_recommendation_popularity(mat_pred, top=None, thresh=0.5, normalize=False)
    5.037037037037036
    """
    # input checks and coerce
    _, _, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=None,
        group_b=None,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize score matrix
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Make matrices binary
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)
    item_count = binary_mat_pred.sum(axis=0)

    val = (binary_mat_pred * item_count).sum(axis=1) / (binary_mat_pred.sum(axis=1))
    return np.nanmean(val)


def mad_score(group_a, group_b, mat_pred, normalize=False):
    r"""
    Mean Absolute Deviation.

    Description
    ----------
    Difference in average score for group_a and group_b.

    Interpretation
    --------------
    A large value of MAD indicates differential treatment of
    group a and group b. A positive value indicates that
    group a received higher scores on average, while
    a negative value indicates higher ratings for group b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        MAD Score : :math:`\texttt{avg_group_a - avg_group_b}`

    References
    ----------
    .. [1] `Ziwei Zhu, Xia Hu, and James Caverlee (2018).
            Fairness-Aware Tensor-Based Recommendation,
            <https://dl.acm.org/doi/pdf/10.1145/3269206.3271795>`
    .. [2] `Y Deldjoo, V W Anelli, H Zamani, A Bellogın, TDi, T D Noia (2021).
            A Flexible Framework for Evaluating User and Item Fairness in Recommender Systems,
            <https://link.springer.com/article/10.1007/s11257-020-09285-1>`
    .. [3] `Y Deldjooa, A Bellogin, T D Noiaa (2021).
            Explaining recommender systems fairness and accuracy through the lens of data characteristics,
            <https://www.sciencedirect.com/science/article/pii/S0306457321001503>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import mad_score
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                            [0.7, 0.9, 0.1, 0.7],
                            [0.3, 0.2, 0.3, 0.3],
                            [0.2, 0.1, 0.7, 0.8],
                            [0.8, 0.7, 0.9, 0.1],
                            [1. , 0.9, 0.3, 0.6],
                            [0.8, 0.9, 0.1, 0.1],
                            [0.2, 0.3, 0.1, 0.5],
                            [0.1, 0.2, 0.7, 0.7],
                            [0.2, 0.7, 0.1, 0.2]])
    >>> mad_score(group_a, group_b, mat_pred, normalize=False)
    0.00833333333333336
    """
    # input checks and coerce
    group_a, group_b, mat_pred, _, _, _, normalize = _recommender_checks(
        group_a=group_a,
        group_b=group_b,
        mat_pred=mat_pred,
        mat_true=None,
        top=None,
        thresh=None,
        normalize=normalize,
    )

    # normalize
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Split by group
    mat_pred_a = mat_pred[group_a == 1]
    mat_pred_b = mat_pred[group_b == 1]

    # Get averages
    avg_a = np.nanmean(mat_pred_a)
    avg_b = np.nanmean(mat_pred_b)

    return avg_a - avg_b


def exposure_l1(group_a, group_b, mat_pred, top=None, thresh=0.5, normalize=False):
    """
    Exposure Total Variation.

    Description
    ----------
    This function computes the total variation norm between the group_a
    exposure distribution to the group_b exposure distribution.

    Interpretation
    --------------
    A total variation divergence of 0 is desired, which occurs when the distributions
    are equal. The maximum value is 1 indicating the distributions are
    very far apart.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Exposure Total Variation

    References
    ----------
    .. [1] `T Giannakas, P Sermpezis, A Giovanidis, T Spyropoulos, G Arvanitakis (2021).
            Fairness in Network-Friendly Recommendations,
            <https://arxiv.org/pdf/2104.00959.pdf>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import exposure_l1
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                            [0.7, 0.9, 0.1, 0.7],
                            [0.3, 0.2, 0.3, 0.3],
                            [0.2, 0.1, 0.7, 0.8],
                            [0.8, 0.7, 0.9, 0.1],
                            [1. , 0.9, 0.3, 0.6],
                            [0.8, 0.9, 0.1, 0.1],
                            [0.2, 0.3, 0.1, 0.5],
                            [0.1, 0.2, 0.7, 0.7],
                            [0.2, 0.7, 0.1, 0.2]])
    >>> exposure_l1(group_a, group_b, mat_pred, top=1, thresh=None, normalize=False)
    0.25
    """
    # input checks and coerce
    group_a, group_b, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=group_a,
        group_b=group_b,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize score matrix
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # Split by group
    mat_pred_a = binary_mat_pred[group_a == 1]
    mat_pred_b = binary_mat_pred[group_b == 1]

    # Get the item exposure distribution for group_a
    item_count_a = mat_pred_a.sum(axis=0)
    item_dist_a = item_count_a / item_count_a.sum()

    # Get the item exposure distribution for group_b
    item_count_b = mat_pred_b.sum(axis=0)
    item_dist_b = item_count_b / item_count_b.sum()

    # Compute Total variation distance
    return 0.5 * mean_absolute_error(item_dist_a, item_dist_b) * len(item_dist_a)


def exposure_kl(group_a, group_b, mat_pred, top=None, thresh=0.5, normalize=False):
    """
    Exposure KL Divergence.

    Description
    ----------
    This function computes the KL divergence from the group_a
    exposure distribution to the group_b exposure distribution.

    Interpretation
    --------------
    A KL divergence of 0 is desired, which occurs when the distributions
    are equal. Higher values of the KL divergence indicate difference
    in exposure distributions of group a and group b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        `KL(exp_min,exp_maj)`

    References
    ----------
    .. [1] `A Dash, A Chakraborty, S Ghosh, A Mukherjee, K P. Gummadi (2021).
            When the Umpire is also a Player: Bias in Private Label Product
            Recommendations on E-commerce Marketplaces,
            <https://arxiv.org/pdf/2102.00141.pdf >`
    .. [2] `T Giannakas, P Sermpezis, A Giovanidis, T Spyropoulos, G Arvanitakis (2021).
            Fairness in Network-Friendly Recommendations,
            <https://arxiv.org/pdf/2102.00141.pdf >`


    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import exposure_kl
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                            [0.7, 0.9, 0.1, 0.7],
                            [0.3, 0.2, 0.3, 0.3],
                            [0.2, 0.1, 0.7, 0.8],
                            [0.8, 0.7, 0.9, 0.1],
                            [1. , 0.9, 0.3, 0.6],
                            [0.8, 0.9, 0.1, 0.1],
                            [0.2, 0.3, 0.1, 0.5],
                            [0.1, 0.2, 0.7, 0.7],
                            [0.2, 0.7, 0.1, 0.2]])
    >>> exposure_kl(group_a, group_b, mat_pred, top=1, thresh=None, normalize=False)
    0.23217831296817806
    """
    # input checks and coerce
    group_a, group_b, mat_pred, _, top, thresh, normalize = _recommender_checks(
        group_a=group_a,
        group_b=group_b,
        mat_pred=mat_pred,
        mat_true=None,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # normalize score matrix
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # Split by group
    mat_pred_a = mat_pred[group_a == 1]
    mat_pred_b = mat_pred[group_b == 1]

    # Get the item exposure distribution for group_a
    binary_mat_pred_a = mat_to_binary(mat_pred_a, top=top, thresh=thresh)
    item_count_a = binary_mat_pred_a.sum(axis=0)
    item_dist_a = item_count_a / item_count_a.sum()

    # Get the item exposure distribution for group_b
    binary_mat_pred_b = mat_to_binary(mat_pred_b, top=top, thresh=thresh)
    item_count_b = binary_mat_pred_b.sum(axis=0)
    item_dist_b = item_count_b / item_count_b.sum()

    # Compute KL divergence between dists
    return entropy(item_dist_a, item_dist_b)


def _recommender_metric_ratio(
    metric, group_a, group_b, mat_pred, mat_true, top=None, thresh=0.5, normalize=False
):
    """
    Metric ratio for recommender systems

    Description
    ----------
    This function computes the ratio of a given metric on minority and majority group.

    Parameters
    ----------
    metric : function
        Metric to compute
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Ratio of metrics : Metric(min)/Metric(maj)
    """
    # input checks and coerce
    group_a, group_b, mat_pred, mat_true, top, thresh, normalize = _recommender_checks(
        group_a=group_a,
        group_b=group_b,
        mat_pred=mat_pred,
        mat_true=mat_true,
        top=top,
        thresh=thresh,
        normalize=normalize,
    )

    # if normalize
    if normalize:
        tens = np.stack((mat_pred, mat_true))
        norm_tens = normalize_tensor(tens)
        mat_pred, mat_true = norm_tens

    # Split by group
    mat_pred_a = mat_pred[group_a == 1]
    mat_pred_b = mat_pred[group_b == 1]
    mat_true_a = mat_true[group_a == 1]
    mat_true_b = mat_true[group_b == 1]

    # compute metrics
    if top is None and thresh is None:
        metric_a = metric(mat_pred_a, mat_true_a)
        metric_b = metric(mat_pred_b, mat_true_b)
    else:
        # metrics that have top and thresh as input
        metric_a = metric(mat_pred_a, mat_true_a, top, thresh)
        metric_b = metric(mat_pred_b, mat_true_b, top, thresh)

    # ratio
    return metric_a / metric_b


def avg_precision_ratio(
    group_a, group_b, mat_pred, mat_true, top=None, thresh=0.5, normalize=False
):
    r"""
    Average precision ratio

    Description
    ----------
    This function computes the ratio of average precision (over users)
    on minority and majority group.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against minority group.
    Higher values show bias against majority group.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Ratio of average precisions : :math:`\frac{\texttt{AVg_precision_min}}{\texttt{AVg_precision_maj}}`

    References
    ----------
    .. [1] `Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang (2021).
            User-oriented Fairness in Recommendation.
            <https://arxiv.org/pdf/2104.10671.pdf>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_precision_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> mat_true = np.array([[0.7, 0.8, 0.4, 0.2],
                          [0.9, 0.9, 0.1, 0.2],
                          [0.3, 0.8, 0.2, 0.6],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.6, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.1, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.1, 0.1, 0.8]])
    >>> avg_precision_ratio(group_a, group_b, mat_pred, mat_true, top=None, thresh=0.2, normalize=False)
    1.161290322580645
    """
    return _recommender_metric_ratio(
        avg_precision, group_a, group_b, mat_pred, mat_true, top, thresh, normalize
    )


def avg_recall_ratio(
    group_a, group_b, mat_pred, mat_true, top=None, thresh=0.5, normalize=False
):
    r"""
    Average recall ratio

    Description
    ----------
    This function computes the ratio of average recall (over users)
    on minority and majority group.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Ratio of average recalls : :math:`\frac{\texttt{AVg_recall_min}}{\texttt{AVg_recall_maj}}`

    References
    ----------
    .. [1] `Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang (2021).
            User-oriented Fairness in Recommendation.
            <https://arxiv.org/pdf/2104.10671.pdf>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_recall_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> mat_true = np.array([[0.7, 0.8, 0.4, 0.2],
                          [0.9, 0.9, 0.1, 0.2],
                          [0.3, 0.8, 0.2, 0.6],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.6, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.1, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.1, 0.1, 0.8]])
    >>> avg_recall_ratio(group_a, group_b, mat_pred, mat_true, top=2, thresh=0.5, normalize=False)
    1.0
    """
    return _recommender_metric_ratio(
        avg_recall, group_a, group_b, mat_pred, mat_true, top, thresh, normalize
    )


def avg_f1_ratio(
    group_a, group_b, mat_pred, mat_true, top=None, thresh=0.5, normalize=False
):
    r"""
    Average f1 ratio

    Description
    ----------
    This function computes the ratio of average f1 (over users)
    on minority and majority group.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against minority group.
    Higher values show bias against majority group.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Ratio of average f1 : :math:`\frac{\texttt{AVg_f1_min}}{\texttt{AVg_f1_maj}}`

    References
    ----------
    .. [1] `Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang (2021).
            User-oriented Fairness in Recommendation.
            <https://arxiv.org/pdf/2104.10671.pdf>`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_f1_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> mat_true = np.array([[0.7, 0.8, 0.4, 0.2],
                          [0.9, 0.9, 0.1, 0.2],
                          [0.3, 0.8, 0.2, 0.6],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.6, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.1, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.1, 0.1, 0.8]])
    >>> avg_f1_ratio(group_a, group_b, mat_pred, mat_true, top=None, thresh=0.5, normalize=False)
    0.9285714285714286
    """

    return _recommender_metric_ratio(
        avg_f1, group_a, group_b, mat_pred, mat_true, top, thresh, normalize
    )


def recommender_rmse_ratio(group_a, group_b, mat_pred, mat_true, normalize=False):
    """
    Recommender RMSE ratio.

    Description
    ----------
    This function computes the ratio of rmse between
    predictions and scores for group_a and group_b.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against group_a.
    Higher values show bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Recommender RMSE ratio : :math:`\frac{\texttt{AVg_rmse_min}}{\texttt{AVg_rmse_maj}}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import recommender_rmse_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> mat_true = np.array([[0.7, 0.8, 0.4, 0.2],
                          [0.9, 0.9, 0.1, 0.2],
                          [0.3, 0.8, 0.2, 0.6],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.6, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.1, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.1, 0.1, 0.8]])
    >>> recommender_rmse_ratio(group_a, group_b, mat_pred, mat_true)
    1.149630441384884
    """
    return _recommender_metric_ratio(
        recommender_rmse,
        group_a,
        group_b,
        mat_pred,
        mat_true,
        top=None,
        thresh=None,
        normalize=normalize,
    )


def recommender_mae_ratio(group_a, group_b, mat_pred, mat_true, normalize=False):
    """
    Recommender MAE ratio.

    Description
    ----------
    This function computes the ratio of mae between
    predictions and scores for group_a and group_b.

    Interpretation
    --------------
    A value of 1 is desired. Lower values show bias against group_a.
    Higher values show bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.

    Returns
    -------
    float
        Recommender MAE ratio : :math:`\frac{\texttt{AVg_mae_min}}{\texttt{AVg_mae_maj}}`

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import recommender_mae_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> mat_pred = np.array([[0.9, 0.8, 0.4, 0.2],
                          [0.7, 0.9, 0.1, 0.7],
                          [0.3, 0.2, 0.3, 0.3],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.8, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.9, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.7, 0.1, 0.2]])
    >>> mat_true = np.array([[0.7, 0.8, 0.4, 0.2],
                          [0.9, 0.9, 0.1, 0.2],
                          [0.3, 0.8, 0.2, 0.6],
                          [0.2, 0.1, 0.7, 0.8],
                          [0.6, 0.7, 0.9, 0.1],
                          [1. , 0.9, 0.3, 0.6],
                          [0.8, 0.1, 0.1, 0.1],
                          [0.2, 0.3, 0.1, 0.5],
                          [0.1, 0.2, 0.7, 0.7],
                          [0.2, 0.1, 0.1, 0.8]])
    >>> recommender_mae_ratio(group_a, group_b, mat_pred, mat_true)
    1.2954545454545452
    """
    return _recommender_metric_ratio(
        recommender_mae,
        group_a,
        group_b,
        mat_pred,
        mat_true,
        top=None,
        thresh=None,
        normalize=normalize,
    )


def recommender_bias_metrics(
    group_a=None,
    group_b=None,
    mat_pred=None,
    mat_true=None,
    top=None,
    thresh=0.5,
    normalize=False,
    metric_type="equal_outcome",
):
    """
    Recommender bias metrics batch computation.

    Description
    ----------
    This function computes all the relevant recommender bias metrics,
    and displays them as a pandas dataframe.

    Parameters
    ----------
    group_a : array-like
        Group membership vector.
    group_b : array-like
        Group membership vector.
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item interaction.
    mat_true : matrix-like
        Matrix with shape (num_users, num_items). A target score
        (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.
    metric_type : 'all', 'item_based', 'equal_outcome' or 'equal_opportunity'
        Specifies which metrics we compute

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    item_perform = {
        "Aggregate Diversity": aggregate_diversity,
        "GINI index": gini_index,
        "Exposure Distribution Entropy": exposure_entropy,
        "Average Recommendation Popularity": avg_recommendation_popularity,
    }

    group_perform = {
        "Mean Absolute Deviation": mad_score,
    }

    group2_perform = {
        "Exposure Total Variation": exposure_l1,
        "Exposure KL Divergence": exposure_kl,
    }

    group_true_perform = {
        "Average Precision Ratio": avg_precision_ratio,
        "Average Recall Ratio": avg_recall_ratio,
        "Average F1 Ratio": avg_f1_ratio,
    }

    group_true_reg_perform = {
        "Recommender RMSE Ratio": recommender_rmse_ratio,
        "Recommender MAE Ratio": recommender_mae_ratio,
    }

    ref_vals = {
        "Aggregate Diversity": 1,
        "GINI index": 0,
        "Exposure Distribution Entropy": "-",
        "Average Recommendation Popularity": "-",
        "Mean Absolute Deviation": 0,
        "Exposure Total Variation": 0,
        "Exposure KL Divergence": 0,
        "Average Precision Ratio": 1,
        "Average Recall Ratio": 1,
        "Average F1 Ratio": 1,
        "Recommender RMSE Ratio": 1,
        "Recommender MAE Ratio": 1,
    }

    item_metrics = []
    out_metrics = []
    opp_metrics = []

    item_metrics += [
        [pf, fn(mat_pred, top, thresh, normalize), ref_vals[pf]]
        for pf, fn in item_perform.items()
    ]
    if group_a is not None:
        out_metrics += [
            [pf, fn(group_a, group_b, mat_pred, normalize), ref_vals[pf]]
            for pf, fn in group_perform.items()
        ]
        out_metrics += [
            [pf, fn(group_a, group_b, mat_pred, top, thresh, normalize), ref_vals[pf]]
            for pf, fn in group2_perform.items()
        ]
    if mat_true is not None:
        opp_metrics += [
            [
                pf,
                fn(group_a, group_b, mat_pred, mat_true, top, thresh, normalize),
                ref_vals[pf],
            ]
            for pf, fn in group_true_perform.items()
        ]
        opp_metrics += [
            [pf, fn(group_a, group_b, mat_pred, mat_true, normalize), ref_vals[pf]]
            for pf, fn in group_true_reg_perform.items()
        ]

    if metric_type == "all":
        metrics = item_metrics + out_metrics + opp_metrics
        return pd.DataFrame(
            metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    if metric_type == "item_based":
        return pd.DataFrame(
            item_metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    if metric_type == "equal_outcome":
        return pd.DataFrame(
            out_metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    if metric_type == "equal_opportunity":
        return pd.DataFrame(
            opp_metrics, columns=["Metric", "Value", "Reference"]
        ).set_index("Metric")

    else:
        raise ValueError(
            "metric_type is not one of : all, item_based, equal_outcome, equal_opportunity"
        )
