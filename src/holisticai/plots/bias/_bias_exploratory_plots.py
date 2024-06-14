# Base Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from holisticai.metrics.bias import confusion_matrix

# utils
from holisticai.utils import get_colors
from holisticai.utils._validation import (
    _check_columns,
    _check_numerical_dataframe,
    _regression_checks,
)


def group_pie_plot(y_feat, ax=None, size=None, title=None):
    """Plot a pie chart showing proportions of groups in a feature.

    Parameters
    ----------
    y_feat : pandas series or numpy array
        Feature of interest
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
    # checks
    if isinstance(y_feat, pd.Series):
        value_counts = y_feat.value_counts()
        labels = value_counts.index.tolist()
    elif isinstance(y_feat, np.ndarray):
        y_feat = pd.Series(y_feat)
        value_counts = y_feat.value_counts()
        labels = value_counts.index.tolist()

    else:
        raise TypeError("input is not a numpy array or pandas series")

    # calculations
    n_b = np.sum(value_counts / np.sum(value_counts) > 0.02)  # noqa: PLR2004
    n_nb = len(value_counts) - n_b
    if n_nb > 0 and n_b > 0:
        labels = [*list(labels[:n_b]), "Others"]
        value_counts = [*list(value_counts[:n_b]), np.sum(value_counts[n_b:])]

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Group proportions")

    # chart
    colors = get_colors(len(value_counts), extended_colors=True)
    hai_palette = sns.color_palette(colors)
    ax.pie(value_counts, labels=labels, colors=hai_palette, autopct="%.0f%%")

    # return
    return ax


def _group_confusion_matrices(group_a, group_b, y_pred, y_true, size=None, title=None):
    """Group confusion matrices comparison.

    Plots three heatmaps showing confusion matrices for
    two groups and the combination of those groups.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Prediction vector (categorical)
    y_true : array-like
        True values (categorical)
    size (optional) : (int, int)
        Size of the figure
    title (optional) : str
        Title of the figure

    Returns
    -------
    matplotlib ax
    """
    # check and coerce
    group_a, group_b, y_pred, y_true, _ = _regression_checks(group_a, group_b, y_pred, y_true, None)

    # Calculating confusion matrices
    a_indices = group_a == 1
    conf_mat_a = confusion_matrix(y_pred[a_indices], y_true[a_indices])
    conf_mat_a = conf_mat_a.to_numpy()
    # normalise on columns
    conf_mat_a = conf_mat_a / conf_mat_a.sum(axis=0)[np.newaxis, :]
    b_indices = group_b == 1
    conf_mat_b = confusion_matrix(y_pred[b_indices], y_true[b_indices])
    conf_mat_b = conf_mat_b.to_numpy()
    # normalise on columns
    conf_mat_b = conf_mat_b / conf_mat_b.sum(axis=0)[np.newaxis, :]
    conf_mat_combi = confusion_matrix(y_pred, y_true)
    conf_mat_combi = conf_mat_combi.to_numpy()
    # normalise on columns
    conf_mat_combi = conf_mat_combi / conf_mat_combi.sum(axis=0)[np.newaxis, :]

    # Setup
    sns.set()
    colors = get_colors(100, extended_colors=True)
    hai_palette = sns.color_palette(colors)
    fig, ax = plt.subplots(1, 3, figsize=size)
    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle("Confusion matrices")

    # charting
    sns.heatmap(
        conf_mat_a,
        annot=True,
        cmap=hai_palette,
        vmin=0,
        vmax=1,
        ax=ax[0],
        cbar=0,
        square=True,
    )
    sns.heatmap(
        conf_mat_b,
        annot=True,
        cmap=hai_palette,
        vmin=0,
        vmax=1,
        ax=ax[1],
        cbar=0,
        square=True,
    )
    sns.heatmap(
        conf_mat_combi,
        annot=True,
        cmap=hai_palette,
        vmin=0,
        vmax=1,
        ax=ax[2],
        cbar=1,
        square=True,
    )
    ax[0].set_title("group_a")
    ax[1].set_title("group_b")
    ax[2].set_title("Both groups")

    # return
    return ax


def distribution_plot(y_feat, p_attr=None, ax=None, size=None, title=None):
    """Plot the Kernel Density Estimation (KDE) distribution of the input data.
    If the protected attribute is included, the distributions are plotted for each
    group independently.

    Parameters
    ----------
    y_feat : pandas series or numpy array (regression)
        Feature of interest
    p_attr : pandas series or numpy array (categorical)
        Protected attribute
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
    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("KDE distribution plot")

    # charting
    colors = get_colors(len(np.unique(p_attr)), extended_colors=True)
    hai_palette = sns.color_palette(colors)
    return sns.kdeplot(x=y_feat, hue=p_attr, fill=True, palette=hai_palette, common_norm=False, ax=ax)


def histogram_plot(y_feat, p_attr=None, ax=None, size=None, title=None):
    """Plot the histogram for categorical data.
    If the protected attribute is included, the histograms are plotted for each
    group independently.

    Parameters
    ----------
    y_feat : pandas series or numpy array (categorical)
        Feature of interest
    p_attr : pandas series or numpy array (categorical)
        Protected attribute
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
    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Histogram Plot")

    colors = get_colors(len(np.unique(p_attr)))
    hai_palette = sns.color_palette(colors)

    if p_attr is None:
        hai_color = hai_palette[0]
        hai_palette = None
    else:
        hai_color = None

    # charting
    sns.histplot(
        x=y_feat,
        hue=p_attr,
        stat="probability",
        fill=True,
        palette=hai_palette,
        color=hai_color,
        common_norm=False,
        ax=ax,
    )
    _, labels = plt.xticks()
    plt.setp(labels, rotation=45)

    # return
    return ax


def correlation_matrix_plot(df, target_feature, n_features=10, cmap="YlGnBu", ax=None, size=None, title=None):
    """Plot the correlation matrix of a given dataframe with respect to
    a given target and a certain number of features.

    Obs. The dataframe must contain only numerical features.

    Parameters
    ----------
    df : (DataFrame)
        Pandas dataframe of the data
    target_feature : (str)
        Column name of the target feature
    n_features (optional) : (int)
        Number of features to plot with the closest correlation to the target
    cmap (optional) : (str)
        Color map to use
    ax (optional) : matplotlib axes
        Pre-existing axes for the plot
    size (optional) : (int, int)
        Size of the figure
    title (optional) : (str)
        Title of the figure

    Returns
    -------
    matplotlib ax
    """
    """Prints the correlation matrix """
    df = _check_numerical_dataframe(df)
    _check_columns(df, target_feature)

    sns.set(font_scale=1.25)
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Correlation matrix")
    corrmat = df.corr()
    cols = corrmat.nlargest(n_features, target_feature)[target_feature].index
    cm = np.corrcoef(df[cols].values.T)
    sns.heatmap(
        cm,
        cbar=False,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=cols.values,
        xticklabels=cols.values,
        cmap=cmap,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return ax
