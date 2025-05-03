import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from holisticai.datasets import Dataset

# utils
from holisticai.utils._validation import (
    _check_columns,
    _check_numerical_dataframe,
)


def correlation_matrix_plot(
    dataset: Dataset,
    features,
    target_feature,
    fixed_features=None,
    n_features=10,
    cmap="YlGnBu",
    ax=None,
    size=None,
    title=None,
):
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

    if target_feature not in features:
        features.append(target_feature)

    if fixed_features is None:
        fixed_features = []

    for f in fixed_features:
        if f not in features:
            features.append(f)

    df = _check_numerical_dataframe(dataset[features])
    _check_columns(df, target_feature)

    sns.set_theme(font_scale=1.25)
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    corrmat = df.corr()
    cols = list(corrmat.nlargest(n_features, target_feature)[target_feature].index)
    for f in fixed_features:
        if f in cols:
            cols.remove(f)
    cols += fixed_features
    cm = np.corrcoef(df[cols].values.T)
    sns.heatmap(
        cm,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=cols,
        xticklabels=cols,
        cmap=cmap,
        ax=ax,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Correlation matrix")
    return ax
