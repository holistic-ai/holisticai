# Base Imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from holisticai.utils import get_colors

# utils
from ...utils._validation import _check_binary, _multiclass_checks

# Import metrics
from ..metrics import frequency_matrix


def frequency_plot(p_attr, y_pred):
    """
    Frequency plot.

    Description
    ----------
    This function plots how frequently members
    of each group fall into each class.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like

    Returns
    -------
    matplotlib ax (or None)
    """
    # check and coerce inputs
    p_attr, y_pred, _, _, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=None,
        classes=None,
    )

    # get success rates
    sr_list = frequency_matrix(p_attr, y_pred, normalize="group")

    # sort by success rate
    sr_tot = sr_list.sum(axis=0) / sr_list.sum(axis=0).sum()
    sr_tot.name = "total"
    sr_list = pd.concat([sr_list, pd.DataFrame(sr_tot).transpose()], axis=0)
    name_classes = sr_list.columns
    n_classes = len(name_classes)

    # charting
    sns.set()

    colors = get_colors(sr_list.shape[0])
    hai_palette = sns.color_palette(colors)

    if n_classes > 2:
        for i in range(n_classes):
            fig, ax = plt.subplots()
            fig.suptitle("Class " + str(name_classes[i]))
            sns.barplot(
                x=sr_list.index.to_list(),
                y=sr_list[name_classes[i]],
                palette=hai_palette,
                ax=ax,
            )
            ax.set_xlabel("Group")
            ax.set_ylabel("Frequency ")
            _, labels = plt.xticks()
            plt.setp(labels, rotation=45)

        return None

    else:
        fig, ax = plt.subplots()
        fig.suptitle("Success Rate Plot (Class {})".format(name_classes[1]))
        sns.barplot(
            x=sr_list.index.to_list(),
            y=sr_list[name_classes[1]],
            palette=hai_palette,
            ax=ax,
        )
        ax.set_xlabel("Group")
        ax.set_ylabel("Frequency")
        _, labels = plt.xticks()
        plt.setp(labels, rotation=45)

        return ax


def statistical_parity_plot(
    p_attr, y_pred, pos_label=1, compare_to=None, ax=None, size=None, title=None
):
    """
    Statistical Parity Plot (Binary Classification).

    Description
    -----------
    This function plots the statistical parity for each group
    along with acceptable bounds. We take the group with maximum
    success rate as the comparison group.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Prediction vector (binary)
    pos_label (optional) : label, default=1
        The positive label
    compare_to (optional) : str or int
        The group we are comparing to
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
    p_attr, y_pred, _, groups, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=None,
        classes=None,
    )

    _check_binary(y_pred, name="y_pred")

    group_dict = dict(zip(groups, range(len(groups))))
    # get success rates.
    sr_list = frequency_matrix(p_attr, y_pred * 1, groups=groups, normalize="group")[
        pos_label
    ].to_numpy()

    # sort by success rate.
    sr_list_sorted, groups_sorted = zip(*sorted(zip(sr_list, groups), reverse=True))
    sr_list_sorted, groups_sorted = list(sr_list_sorted), list(groups_sorted)

    # statistical parity list
    if compare_to is not None:
        sp_list = sr_list_sorted - sr_list[group_dict[compare_to]]
    else:
        sp_list = sr_list_sorted - np.max(sr_list)

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Statistical Parity plot")

    # charting
    colors = get_colors(len(groups))
    hai_palette = sns.color_palette(colors)
    ax.set_xlabel("Group")
    ax.set_ylabel("Statistical Parity")
    sns.barplot(x=groups_sorted, y=sp_list, palette=hai_palette, ax=ax)
    # horizontal lines
    ax.axhline(y=-0.1, color="grey", linestyle="--", label="lower bound")
    ax.axhline(y=0.1, color="grey", linestyle="--", label="upper bound")
    ax.axhspan(-0.1, 0.1, alpha=0.3, color="grey", zorder=0, label="fair area")
    # tilt labels
    _, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    ax.legend()

    return ax


def disparate_impact_plot(
    p_attr, y_pred, pos_label=1, compare_to=None, ax=None, size=None, title=None
):
    """
    Disparate Impact Plot (Binary Classification).

    Description
    -----------
    This function plots the disparate impact for each group
    along with acceptable bounds. We take the group with maximum
    success rate as the 'majority group'.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Prediction vector
    pos_label : label, default=1
        The positive label
    compare_to (optional) : str or int
        The group we are comparing to
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
    p_attr, y_pred, _, groups, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=None,
        classes=None,
    )

    _check_binary(y_pred, name="y_pred")

    group_dict = dict(zip(groups, range(len(groups))))
    # get success rates
    sr_list = frequency_matrix(p_attr, 1 * y_pred, groups)[pos_label].to_numpy()

    # sort by success rate.
    sr_list_sorted, groups_sorted = zip(*sorted(zip(sr_list, groups), reverse=True))
    sr_list_sorted, groups_sorted = list(sr_list_sorted), list(groups_sorted)

    # disparate impact list
    if compare_to is not None:
        di_list = sr_list_sorted / sr_list[group_dict[compare_to]]
    else:
        di_list = sr_list_sorted / sr_list_sorted[0]

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Disparate Impact plot")

    # charting
    colors = get_colors(len(groups))
    hai_palette = sns.color_palette(colors)
    ax.set_xlabel("Group")
    ax.set_ylabel("Disparate Impact")
    sns.barplot(x=groups_sorted, y=di_list, palette=hai_palette, ax=ax)
    # horizontal lines
    ax.axhspan(0.8, 1.2, alpha=0.3, color="grey", label="fair area")
    ax.axhline(y=1.2, color="grey", linestyle="--", label="upper bound")
    ax.axhline(y=0.8, color="grey", linestyle="--", label="lower bound")
    # tilt labels
    _, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    # legend
    ax.legend()

    # return
    return ax


def frequency_matrix_plot(
    p_attr,
    y_pred,
    groups=None,
    classes=None,
    normalize=None,
    reverse_colors=False,
    ax=None,
    size=None,
    title=None,
):
    """
    Frequency Matrix Plot.

    Description
    -----------
    This function plots the matrix of occurence rate (count)
    for each group, class pair. We include the option to normalise
    over groups or classes.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Prediction vector (categorical)
    groups (optional) : array or list
        The groups in order
    classes (optional) : array or list
        The classes in order
    normalize (optional): None, 'group' or 'class'
        According to which of group or class we normalize
    reverse_colors (optional): bool, default=False
        Option to reverse the color palette
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
    p_attr, y_pred, _, groups, classes = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=None,
        groups=groups,
        classes=classes,
    )

    # compute frequency matrix
    colors = get_colors(10, extended_colors=True, reverse=reverse_colors)
    hai_palette = sns.color_palette(colors)
    sr_mat = frequency_matrix(
        p_attr, y_pred, groups=groups, classes=classes, normalize=normalize
    )

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Frequency matrix plot")

    # charting
    if normalize is None:
        sns.heatmap(sr_mat, annot=True, cmap=hai_palette, ax=ax)
    else:
        sns.heatmap(sr_mat, annot=True, fmt=".2%", cmap=hai_palette, ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Group")

    # return
    return ax


def accuracy_bar_plot(p_attr, y_pred, y_true, ax=None, size=None, title=None):
    """
    Accuracy Bar Plot.

    Description
    -----------
    This function plots the accuracy of the predictions
    for each group.

    Parameters
    ----------
    p_attr : array-like
        Protected attribute vector
    y_pred : array-like
        Prediction vector
    y_true : array-like
        Target vector
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
    p_attr, y_pred, y_true, groups, _ = _multiclass_checks(
        p_attr=p_attr,
        y_pred=y_pred,
        y_true=y_true,
        groups=None,
        classes=None,
    )

    # loop over groups
    acc_list = []
    for c in groups:
        members = p_attr == c
        pred_c = y_pred[members]
        true_c = y_true[members]
        truepred = (pred_c == true_c).sum()
        acc = truepred / len(pred_c)
        acc_list.append(acc)

    acc_list = list(acc_list)
    groups = list(groups)

    # tot
    acc_tot = (y_pred == y_true).sum() / len(y_pred)
    acc_list.append(acc_tot)
    groups.append("Total")

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Accuracy Bar Plot")

    # charting
    colors = get_colors(len(groups))
    hai_palette = sns.color_palette(colors)
    ax.set_xlabel("Group")
    ax.set_ylabel("Accuracy")
    sns.barplot(x=groups, y=acc_list, palette=hai_palette, ax=ax)

    return ax
