# Base Imports
import numpy as np
import seaborn as sns

# utils
from holisticai.utils import get_colors, mat_to_binary, normalize_tensor
from holisticai.utils._validation import _recommender_checks
from matplotlib import pyplot as plt


def long_tail_plot(mat_pred, top=None, thresh=0.5, normalize=False, ax=None, size=None, title=None):
    """
    Long Tail Plot.

    Description
    -----------
    This function plots the counts in the predictions
    for all items.

    Parameters
    ----------
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.
    ax (optional) : matplotlib axes
        Pre-existing axes for the plot
    size (optional) : (int, int)
        Size of the figure
    title (optional) : str
        Title of the figure

    Returns
    -------
    matplotlib ax
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

    # normalize
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # make binary (ie shown / not shown)
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # item counts and sort
    item_counts = list(binary_mat_pred.sum(axis=0))
    item_counts_sorted = sorted(item_counts, reverse=True)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # chart
    ax.set_xlabel("Items (sorted by popularity)")
    ax.set_ylabel("Item Count")
    hai_color = get_colors(1, extended_colors=False)
    ax.plot(
        range(len(item_counts_sorted)),
        item_counts_sorted,
        linewidth=2,
        color=hai_color[0],
    )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Long Tail Plot")

    return ax


def exposure_diff_plot(
    group_a,
    group_b,
    mat_pred,
    top=None,
    thresh=0.5,
    normalize=False,
    ax=None,
    size=None,
    title=None,
):
    """
    Exposure Difference plot.

    Description
    -----------
    This function plots the difference in the exposure
    distributions between group_a and group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold in (0,1) range indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.
    ax (optional) : matplotlib axes
        Pre-existing axes for the plot
    size (optional) : (int, int)
        Size of the figure
    title (optional) : str
        Title of the figure

    Returns
    -------
    matplotlib ax
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

    # normalise
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # make binary (ie shown / not shown)
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # Split by group
    mat_pred_a = binary_mat_pred[group_a == 1]
    mat_pred_b = binary_mat_pred[group_b == 1]

    # Get the item exposure distribution for min
    item_count_a = np.nansum(mat_pred_a, axis=0)
    item_dist_a = item_count_a / item_count_a.sum()

    # Get the item exposure distribution for maj
    item_count_b = np.nansum(mat_pred_b, axis=0)
    item_dist_b = item_count_b / item_count_b.sum()

    # take difference
    item_dist_diff = item_dist_a - item_dist_b

    # sort
    item_dist_diff_sorted = sorted(item_dist_diff, reverse=True)
    item_dist_diff_sorted = list(item_dist_diff_sorted)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # chart
    ax.set_xlabel("Items (sorted by exposure difference)")
    ax.set_ylabel("$Exposure_a - Exposure_b$")
    hai_color = get_colors(1, extended_colors=False)
    ax.plot(
        range(len(item_dist_diff_sorted)),
        item_dist_diff_sorted,
        linewidth=2,
        color=hai_color[0],
    )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Exposure Difference Plot")

    return ax


def exposure_ratio_plot(
    group_a,
    group_b,
    mat_pred,
    top=None,
    thresh=0.5,
    normalize=False,
    ax=None,
    size=None,
    title=None,
):
    """
    Exposure Ratio plot.

    Description
    -----------
    This function plots the ratio in the exposure
    distributions between group_a and group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    mat_pred : matrix-like
        Matrix with shape (num_users, num_items). A recommender
        score (binary or soft pred) for each user,item pair.
    top (optional) : int
        If not None, the number of items that are shown to each user.
    thresh (optional) : float
        Threshold in (0,1) range indicating value at which
        a given item is shown to user (if top is None).
    normalize (optional) : bool
        If True, normalises the data matrix to [0,1] range.
    ax (optional) : matplotlib axes
        Pre-existing axes for the plot
    size (optional) : (int, int)
        Size of the figure
    title (optional) : str
        Title of the figure

    Returns
    -------
    matplotlib ax
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

    # normalise
    if normalize:
        mat_pred = normalize_tensor(mat_pred)

    # make binary (ie shown / not shown)
    binary_mat_pred = mat_to_binary(mat_pred, top=top, thresh=thresh)

    # Split by group
    mat_pred_a = binary_mat_pred[group_a == 1]
    mat_pred_b = binary_mat_pred[group_b == 1]

    # Get the item exposure distribution for min
    item_count_a = np.nansum(mat_pred_a, axis=0)
    item_dist_a = item_count_a / item_count_a.sum()

    # Get the item exposure distribution for maj
    item_count_b = np.nansum(mat_pred_b, axis=0)
    item_dist_b = item_count_b / item_count_b.sum()

    # take ratio
    item_dist_rat = item_dist_a / item_dist_b

    # sort by absolute value
    item_dist_rat_sorted = sorted(item_dist_rat, reverse=True)
    item_dist_rat_sorted = list(item_dist_rat_sorted)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # chart
    ax.set_xlabel("Items (sorted by exposure ratio)")
    ax.set_ylabel("$Exposure_a/Exposure_b$")
    hai_color = get_colors(1, extended_colors=False)
    ax.plot(
        range(len(item_dist_rat_sorted)),
        item_dist_rat_sorted,
        linewidth=2,
        color=hai_color[0],
    )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Exposure Ratio Plot")

    return ax
