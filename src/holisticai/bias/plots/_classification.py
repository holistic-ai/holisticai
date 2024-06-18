# Base Imports
import numpy as np
import seaborn as sns

# utils
from holisticai.utils import get_colors
from holisticai.utils._validation import _check_binary, _classification_checks
from matplotlib import pyplot as plt

# sklearn imports
from sklearn.metrics import roc_curve


def abroca_plot(group_a, group_b, y_pred, y_true, ax=None, size=None, title=None):
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
    y_pred : array-like
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

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from holisticai.datasets import load_dataset
    >>> from holisticai.bias.plots import abroca_plot
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    >>> group_a = np.random.randint(0, 2, 1000)
    >>> group_b = np.random.randint(0, 2, 1000)
    >>> y_pred = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    >>> abroca_plot(group_a, group_b, y_pred, y)
    """
    # check and coerce
    group_a, group_b, y_pred, y_true = _classification_checks(group_a, group_b, y_pred, y_true)
    _check_binary(y_true, "y_true")

    # split data by groups
    y_true_a = y_true[group_a == 1]
    y_pred_a = y_pred[group_a == 1]
    y_true_b = y_true[group_b == 1]
    y_pred_b = y_pred[group_b == 1]
    fpr_a, tpr_a, _ = roc_curve(y_true_a, y_pred_a, pos_label=1)
    fpr_b, tpr_b, _ = roc_curve(y_true_b, y_pred_b, pos_label=1)

    # setup
    sns.set_theme()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

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
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Abroca plot")

    # return
    return ax
