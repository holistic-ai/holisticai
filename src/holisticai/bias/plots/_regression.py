# Base Imports
import numpy as np
import seaborn as sns

# utils
from holisticai.utils import get_colors
from holisticai.utils._validation import _multiclass_checks, _regression_checks
from matplotlib import pyplot as plt

# sklearn imports
from sklearn.metrics import mean_absolute_error, mean_squared_error


def success_rate_curve(group_a, group_b, y_pred, ax=None, size=None, title=None):
    """
    Success rate A vs B curve.

    Description
    -----------
    This function plots the group_a pass rate
    vs group_b pass rate curve.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
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
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # calculation
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    colors = get_colors(1)
    ax.plot(np.array(pass_b) * 100.0, np.array(pass_a) * 100.0, color=colors[0])
    ax.plot([0, 100], [0, 100], linestyle="--", label="ideal curve", color="grey")
    ax.set_xlabel("Group B Success Rate %")
    ax.set_ylabel("Group A Success Rate %")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Sucess rate A vs B curve")

    return ax


def statistical_parity_curve(group_a, group_b, y_pred, x_axis="score", ax=None, size=None, title=None):
    """
    Statistical Parity Curve

    Description
    -----------
    This function plots the statistical parity versus
    threshold curves for min and maj groups.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    x_axis (optional) : 'score' or 'quantile'
        Indicates whether the x axis is the scores
        or data quantiles
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
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # calculations
    thresh = np.linspace(0, 1, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()

    # setup
    sns.set_theme()
    colors = get_colors(1)
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    if x_axis == "score":
        ax.plot(
            pass_value,
            pass_a - pass_b,
            color=colors[0],
            label="statistical parity curve",
        )
        ax.plot(
            [np.min(pass_value), np.max(pass_value)],
            [0.1, 0.1],
            linestyle="--",
            label="upper bound",
            color="grey",
        )
        ax.plot(
            [np.min(pass_value), np.max(pass_value)],
            [-0.1, -0.1],
            linestyle="--",
            label="lower bound",
            color="grey",
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Success Rate A - Sucess Rate B")
        ax.legend()
    elif x_axis == "quantile":
        ax.plot(thresh, pass_a - pass_b, color=colors[0], label="statistical parity curve")
        ax.plot([0, 1], [0.1, 0.1], linestyle="--", label="upper bound", color="grey")
        ax.plot([0, 1], [-0.1, -0.1], linestyle="--", label="lower bound", color="grey")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Success Rate A - Sucess Rate B")
        ax.legend()

    else:
        msg = "x_axis is not one of : quantile, score"
        raise ValueError(msg)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Statistical Parity Curve")

    return ax


def disparate_impact_curve(group_a, group_b, y_pred, x_axis="score", ax=None, size=None, title=None):
    """
    Disparate Impact curve

    Description
    -----------
    This function plots the Disparate Impact versus threshold curve.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Predictions vector (regression)
    x_axis (optional) : 'score' or 'quantile'
        Indicates whether the x axis is the scores
        or data quantiles
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
    # check and coerce inputs
    group_a, group_b, y_pred, _, _ = _regression_checks(group_a, group_b, y_pred, None, None)

    # calculations
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    pass_a = y_binary[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_binary[group_b == 1].sum(axis=0) / group_b.sum()

    # setup
    sns.set_theme()
    colors = get_colors(1)
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    if x_axis == "score":
        ax.plot(pass_value, pass_a / pass_b, color=colors[0], label="disparate impact curve")
        ax.plot(
            [np.min(pass_value), np.max(pass_value)],
            [0.8, 0.8],
            linestyle="--",
            color="grey",
            label="upper bound",
        )
        ax.plot(
            [np.min(pass_value), np.max(pass_value)],
            [1.2, 1.2],
            linestyle="--",
            color="grey",
            label="lower bound",
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Disparate Impact")
        ax.legend()

    elif x_axis == "quantile":
        ax.plot(thresh, pass_a / pass_b, color=colors[0], label="disparate impact curve")
        ax.plot([0, 1], [0.8, 0.8], linestyle="--", color="grey", label="upper bound")
        ax.plot([0, 1], [1.2, 1.2], linestyle="--", color="grey", label="lower bound")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Disparate Impact")
        ax.legend()

    else:
        msg = "x_axis is not one of : score, quantile"
        raise ValueError(msg)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Disparate Impact Curve")

    return ax


def success_rate_curves(p_attr, y_pred, groups=None, x_axis="score", ax=None, size=None, title=None):
    """
    Success Rate Curve

    Description
    -----------
    This function plots the success rate vs threshold curve
    for each group.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Predictions vector (regression)
    groups (optional) : list
        The groups we are considering
    x_axis (optional) : 'score' or 'quantile'
        Indicates whether the x axis is the scores
        or data quantiles
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
    # check and coerce inputs
    p_attr, y_pred, _, _, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=None,
        classes=None,
    )

    # calculations
    thresh = np.linspace(1, 0, 150)
    pass_value = np.quantile(y_pred, thresh)
    y_binary = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)
    if groups is None:
        groups = np.sort(np.unique(p_attr))

    # setup
    colors = get_colors(len(groups) + 1, extended_colors=True)
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    if x_axis == "score":
        for i, c in enumerate(groups):
            members = p_attr == c
            pass_c = y_binary[members].sum(axis=0) / members.sum()
            ax.plot(
                pass_value,
                np.array(pass_c) * 100.0,
                label="SR_" + str(c),
                color=colors[i + 1],
            )
        # tot plot
        pass_tot = y_binary.sum(axis=0) / len(p_attr)
        ax.plot(pass_value, np.array(pass_tot) * 100.0, label="SR_tot", color="grey")
        ax.set_xlabel("Score")
        ax.legend()

    elif x_axis == "quantile":
        for i, c in enumerate(groups):
            members = p_attr == c
            pass_c = y_binary[members].sum(axis=0) / members.sum()
            ax.plot(
                thresh,
                np.array(pass_c) * 100.0,
                label="SR_" + str(c),
                color=colors[i + 1],
            )
        # tot plot
        pass_tot = y_binary.sum(axis=0) / len(p_attr)
        ax.plot(thresh, np.array(pass_tot) * 100.0, label="SR_tot", color="grey")
        ax.set_xlabel("Quantile")
        ax.legend()

    else:
        msg = "x_axis is not one of : score, quantile"
        raise ValueError(msg)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Success Rate Curves")

    return ax


def rmse_bar_plot(p_attr, y_pred, y_true, ax=None, size=None, title=None):
    """
    RMSE Bar Plot

    Description
    -----------
    This function plots the RMSE for each group
    as a bar plot.


    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Predictions vector (regression)
    y_true : array-like
        Target vector (regression)
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
    # check and coerce inputs
    p_attr, y_pred, y_true, _, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=None,
        classes=None,
    )

    # extract groups
    groups = np.sort(np.unique(p_attr))
    rmse_list = []

    # loop over groups
    for c in groups:
        members = p_attr == c
        pred_c = y_pred[members]
        true_c = y_true[members]
        rmse_list.append(mean_squared_error(true_c, pred_c, squared=False))
    # tot
    rmse_tot = mean_squared_error(y_true, y_pred, squared=False)
    rmse_list.append(rmse_tot)
    groups = list(groups)
    groups.append("Total")

    # sort
    rmse_list_sorted, groups_sorted = zip(*sorted(zip(rmse_list, groups)))
    rmse_list_sorted, groups_sorted = list(rmse_list_sorted), list(groups_sorted)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    colors = get_colors(len(groups))
    hai_palette = sns.color_palette(colors)
    ax.set_xlabel("Group")
    ax.set_ylabel("RMSE")
    sns.barplot(x=groups_sorted, y=rmse_list_sorted, palette=hai_palette)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=45)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("RMSE bar plot")

    return ax


def mae_bar_plot(p_attr, y_pred, y_true, ax=None, size=None, title=None):
    """
    MAE Bar Plot

    Description
    -----------
    This function plots the MAE for each group
    as a bar plot.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Predictions vector (regression)
    y_true : array-like
        Target vector (regression)
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
    # check and coerce inputs
    p_attr, y_pred, y_true, _, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=None,
        classes=None,
    )

    # extract groups
    groups = np.sort(np.unique(p_attr))
    mae_list = []

    # loop over groups
    for c in groups:
        members = p_attr == c
        pred_c = y_pred[members]
        true_c = y_true[members]
        mae_list.append(mean_absolute_error(true_c, pred_c))

    # tot
    mae_tot = mean_absolute_error(y_true, y_pred)
    mae_list.append(mae_tot)
    groups = list(groups)
    groups.append("Total")

    # sort
    mae_list_sorted, groups_sorted = zip(*sorted(zip(mae_list, groups)))
    mae_list_sorted, groups_sorted = list(mae_list_sorted), list(groups_sorted)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # charting
    colors = get_colors(len(groups))
    hai_palette = sns.color_palette(colors)
    ax.set_xlabel("Group")
    ax.set_ylabel("MAE")
    sns.barplot(x=groups_sorted, y=mae_list_sorted, palette=hai_palette)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("MAE bar plot")

    return ax
