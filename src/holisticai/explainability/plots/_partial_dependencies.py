import matplotlib.pyplot as plt
import numpy as np
from holisticai.explainability.metrics.global_importance._xai_ease_score import XAIEaseAnnotator
from holisticai.utils import Importances, PartialDependence


def plot_partial_dependence(
    partial_dependence: PartialDependence,
    ranked_feature_importance: Importances,
    subplots=(1, 1),
    figsize=None,
    class_idx=0,
):
    """
    Plots the partial dependence of features on the predicted target.

    Parameters
    ----------
    partial_dependence: PartialDependence
            The partial dependence object containing the computed partial dependence values.
    ranked_feature_importance: RankedFeatureImportance
        The ranked feature importance object containing the feature names and their importance scores.
    subplots: (tuple, optional)
        The shape of the subplots grid. Defaults to (1, 1).
    figsize: (tuple, optional)
        The size of the figure. Defaults to None.

    Returns
    -------
        fig: The matplotlib figure object containing the plot.

    Example
    -------
    >>> partial_dependence = PartialDependence(values=[...])
    >>> ranked_feature_importance = Importances(values=[...], feature_names=[...])
    >>> plot_partial_dependence(partial_dependence, ranked_feature_importance)

    The plot should look like this:

    .. image:: /_static/images/xai_plot_partial_dependence.png
        :alt: Plot Partial Dependence
    """
    partial_dependence_values = partial_dependence.values[class_idx]
    _, axs = plt.subplots(*subplots, figsize=figsize)
    axs = [axs] if isinstance(axs, plt.Axes) else axs.flatten()
    n_plots = min(len(axs), len(partial_dependence_values))
    annotator = XAIEaseAnnotator()
    for feature_index in range(n_plots):
        ax = axs[feature_index]
        individuals = partial_dependence_values[feature_index]["individual"][0]
        average = partial_dependence_values[feature_index]["average"][0]
        x = partial_dependence_values[feature_index]["grid_values"][0]
        level = annotator.compute_xai_ease_score_data(partial_dependence_values, ranked_feature_importance).set_index(
            "feature"
        )["scores"]
        feature_name = ranked_feature_importance.feature_names[feature_index]
        feature_value = ranked_feature_importance[feature_index]

        curve_media = np.mean(individuals, axis=0)
        curve_std = np.std(individuals, axis=0)
        curve_lower = curve_media - curve_std
        curve_upper = curve_media + curve_std

        ax.plot(x, average, color="blue", label=level.loc[feature_name])
        # for curve in individuals:
        #    ax.plot(x, curve, alpha=0.05, color="skyblue")
        ax.fill_between(x, curve_lower, curve_upper, color="skyblue", alpha=0.2)

        ymin = individuals.min()
        ymax = individuals.max()
        ax.set_ylim(ymin, ymax)

        xmin = x.min()
        xmax = x.max()
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Predicted Target")

        ax.set_title(f"{feature_name} ({feature_value:.3f})")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
