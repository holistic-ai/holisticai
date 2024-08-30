import numpy as np
import pandas as pd
from holisticai.explainability.metrics.local_importance import compute_importance_distribution
from holisticai.utils import Importances
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon


def plot_feature_importance(feature_importance: Importances, alpha=0.8, top_n=20, ax=None):
    """
    Bar plot of ranked feature importance.

    Parameters
    ----------
    feature_importance: Importances
        The feature importance data.
    top_n: (int, optional)
        The number of top features to display. Defaults to 20.
    alpha: (float, optional)
        Percentage of importance to consider as top features. Defaults to 0.8.
    ax: (matplotlib.axes.Axes, optional)
        The matplotlib axes to plot on. If not provided, a new figure and axes will be created.

    Returns
    -------
        matplotlib.axes.Axes: The matplotlib axes object containing the plot.

    Example
    -------
    >>> feature_importance = Importances(
    ...     values=np.array([0.1, 0.2, 0.3, 0.4]), feature_names=["A", "B", "C", "D"]
    ... )
    >>> plot_feature_importance(feature_importance)

    The plot should look like this:

    .. image:: /_static/images/xai_plot_feature_importance.png
        :alt: Plot Feature Importance

    """

    ranked_feature_importance = feature_importance.top_alpha(alpha=alpha)
    ranked_feature_importance = ranked_feature_importance.as_dataframe().set_index("Variable")
    feature_importances = feature_importance.as_dataframe().set_index("Variable")

    feature_importances.loc[:, "color"] = "#21918C"
    feature_importances.loc[ranked_feature_importance.index, "color"] = "#440154"
    feature_importances.reset_index(inplace=True, drop=False)

    top_n = min(top_n, len(feature_importances))
    top_features = feature_importances.sort_values(by="Importance", ascending=True).tail(top_n)
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax = top_features.plot(kind="barh", x="Variable", y="Importance", color=top_features["color"], legend=False, ax=ax)
    ax.axhline(y=len(top_features) - len(ranked_feature_importance) - 0.5, color="red", linestyle="--", linewidth=2)
    ax.grid()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")

    if hasattr(feature_importance, "strategy"):
        ax.set_title(f"{feature_importance.strategy.title()} Feature Importance")
    else:
        ax.set_title("Feature Importance")

    return ax


def plot_local_importance_distribution(local_importances, ax=None, k=5, num_samples=10000, random_state=42, **kargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    densities = compute_importance_distribution(
        local_importances, k=k, num_samples=num_samples, random_state=random_state
    )
    ax.hist(densities, bins=50, histtype="step", linewidth=1.5, **kargs)
    ax.set_title("Probability Distribution (Histogram Outline)")
    ax.set_xlabel("Feature Importance Entropy")
    ax.set_ylabel("Frequency")
    ax.grid()
    ax.legend()


def plot_predictions_vs_interpretability(y_score, local_importances, ax=None, **kargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    total_importances = local_importances.data["DataFrame"].to_numpy()
    num_features = total_importances.shape[1]
    feature_equal_weight = np.array([1.0 / num_features] * num_features)

    spread = pd.Series([jensenshannon(i, feature_equal_weight, base=2) for i in total_importances])

    ax.scatter(y_score, spread, alpha=0.3, **kargs)

    ax.grid(True)
    ax.set_xlabel("Ouput Probability")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("Higher value means more interpretability")
