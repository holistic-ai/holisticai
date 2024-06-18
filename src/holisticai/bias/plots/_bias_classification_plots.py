# Base Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# utils
from holisticai.utils import get_colors
from holisticai.utils._validation import _check_binary, _regression_checks

# sklearn imports
from sklearn.metrics import roc_curve


def abroca_plot(group_a, group_b, y_score, y_true, ax=None, size=None, title=None):
    """
    Abroca plot

    Description
    -----------
    This function plots the roc curve for both groups
    revealing the area between them (abroca).

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
    # check and coerce
    group_a, group_b, y_score, y_true, _ = _regression_checks(group_a, group_b, y_score, y_true, None)
    _check_binary(y_true, "y_true")

    # split data by groups
    y_true_a = y_true[group_a == 1]
    y_score_a = y_score[group_a == 1]
    y_true_b = y_true[group_b == 1]
    y_score_b = y_score[group_b == 1]
    fpr_a, tpr_a, _ = roc_curve(y_true_a, y_score_a, pos_label=1)
    fpr_b, tpr_b, _ = roc_curve(y_true_b, y_score_b, pos_label=1)

    # setup
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("Abroca plot")

    # charting
    colors = get_colors(2)
    plt.plot(fpr_a, tpr_a, label="roc curve group a", color=colors[0])
    plt.plot(fpr_b, tpr_b, label="roc curve group b", color=colors[1])
    plt.fill(
        np.append(fpr_a, fpr_b[::-1]),
        np.append(tpr_a, tpr_b[::-1]),
        color="grey",
        alpha=0.3,
    )
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.legend()

    # return
    return ax
