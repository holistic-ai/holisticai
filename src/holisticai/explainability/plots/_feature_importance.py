import numpy as np
import pandas as pd
import seaborn as sns
from holisticai.explainability.metrics.global_feature_importance import fluctuation_ratio
from holisticai.explainability.metrics.local_feature_importance import (
    compute_importance_distribution,
    importance_stability,
    local_normalized_desviation,
    rank_consistency,
)
from holisticai.utils import Importances
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


"""
def create_metric_table(partial_dependencies, importances):
    top_fluctuation_ratios = fluctuation_ratio(partial_dependencies, importances, top_n=top_n, aggregated=False)
    df = importances.as_dataframe()
    df['Fluctuation Ratio'] = top_fluctuation_ratios
    return df
"""


def plot_top_explainable_global_feature_importances(partial_dependencies, importances, model_name, top_n):
    fr_df = fluctuation_ratio(partial_dependencies, importances, top_n=top_n, aggregated=False)

    df = importances.as_dataframe().set_index("Variable")

    df = (
        pd.concat([df, fr_df], axis=1)
        .dropna()
        .sort_values("Importance", ascending=False)
        .reset_index()
        .rename({"index": "Variable"}, axis=1)
    )
    score = fluctuation_ratio(partial_dependencies, importances, top_n=top_n)

    if top_n is not None:
        df = df.iloc[:top_n]

    base_color = "#4A6BC1"
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.barh(np.arange(len(df)) - 0.15, df["Importance"], height=0.3, color=base_color, alpha=0.8)

    # Add oscillation markers
    feature_names = [f.rsplit("_")[-1] for f in df["Variable"].tolist()]

    # Customize the plot
    plt.yticks(range(len(df)), feature_names)
    plt.xlabel("Value", fontsize=12)

    # Add a second x-axis for oscillation
    plt.gca().invert_yaxis()
    ax1 = plt.gca()

    ax2 = ax1.twiny()
    ax2.barh(np.arange(len(df)) + 0.15, df["Fluctuation Ratio"], height=0.3, color="#47B39C", alpha=0.8)
    ax2.set_xlim(0, 1)

    # Set labels and titles
    ax1.set_xlabel("Permutation Feature Importance", color=base_color, fontsize=12)
    ax2.set_xlabel("Fluctuation Ratio", color="#47B39C", fontsize=12)

    plt.title(f"{model_name} [FR={score:.3f}]", fontsize=14, pad=20)
    ax1.tick_params(axis="x", colors=base_color)
    ax2.tick_params(axis="x", colors="#47B39C")

    ax2.grid(True)


def plot_local_feature_importances_stability(local_importances, top_n=None, model_name=None):
    local_importances_values = np.abs(local_importances.values)
    local_importances_values /= local_importances_values.sum(axis=1, keepdims=True)
    avg_importances = local_importances_values.mean(axis=0)

    feature_names = local_importances.feature_names

    df = pd.DataFrame({"Variable": feature_names, "Importance": avg_importances})  # .set_index('Variable')
    df["importance_stability"] = np.array(importance_stability(local_importances_values, aggregate=False))
    df = df.sort_values("Importance", ascending=False).reset_index().dropna()

    score = importance_stability(local_importances_values, aggregate=True)
    if top_n is not None:
        df = df.iloc[:top_n]

    base_color = "#4A6BC1"
    base_color2 = "#C14A6B"
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.barh(np.arange(len(df)) - 0.15, df["Importance"], height=0.3, color=base_color, alpha=0.8)

    # Add oscillation markers
    feature_names = [f.rsplit("_")[-1] for f in df["Variable"].tolist()]

    # Customize the plot
    plt.yticks(range(len(df)), feature_names)
    plt.xlabel("Value", fontsize=12)

    # Add a second x-axis for oscillation
    plt.gca().invert_yaxis()
    ax1 = plt.gca()

    ax2 = ax1.twiny()
    ax2.barh(np.arange(len(df)) + 0.15, df["importance_stability"], height=0.3, color=base_color2, alpha=0.8)
    ax2.set_xlim(0, 1)

    # Set labels and titles
    ax1.set_xlabel("SHAP Importance", color=base_color, fontsize=12)
    ax2.set_xlabel("Importance Stability", color=base_color2, fontsize=12)

    plt.title(f"{model_name} [FR={score:.3f}]", fontsize=14, pad=20)
    ax1.tick_params(axis="x", colors=base_color)
    ax2.tick_params(axis="x", colors=base_color2)

    ax2.grid(True)


def plot_ranking_consistency(local_importances, model_name):
    base_color = "#5B7BE9"
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ffffff", base_color])
    local_importances_values = local_importances.values
    values = local_normalized_desviation(local_importances_values)
    all_scores = rank_consistency(local_importances_values, aggregate=False)
    score = rank_consistency(local_importances.values)
    indexes = np.argsort(all_scores)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    sns.heatmap(values[:, indexes], cmap=cmap, cbar=True, yticklabels=False)
    title = f"{model_name} [RC={score:.3f}]"
    plt.title(title)
    plt.ylabel("Samples")
    plt.xlabel("Features")
    rect = patches.Rectangle(
        (0, 0), 1, 1, linewidth=1, edgecolor="black", facecolor="none", transform=plt.gca().transAxes
    )
    plt.gca().add_patch(rect)
