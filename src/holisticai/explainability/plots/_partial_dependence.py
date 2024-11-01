import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from holisticai.explainability.metrics.global_feature_importance._fluctuation_ratio import fluctuation_ratio
from holisticai.explainability.metrics.global_feature_importance._xai_ease_score import XAIEaseAnnotator
from holisticai.utils import Importances, PartialDependence
from matplotlib import cm
from scipy.interpolate import interp1d


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


def get_oscillations_from_individuals(grid_values, individuals):
    X = grid_values
    indice_oscilacion_normalizados = []
    for Y in individuals:
        interpolacion = interp1d(X, Y, kind="linear")
        X_nuevo = np.linspace(min(X), max(X), 20)
        Y_nuevo = interpolacion(X_nuevo)
        derivada = np.diff(Y_nuevo)
        cambios_signo = np.sum(np.diff(np.sign(derivada)) != 0)
        indice_oscilacion_normalizados.append(cambios_signo / len(Y_nuevo))
    return indice_oscilacion_normalizados


# Function to plot the explainable partial dependence with oscillations and importance visualization
def plot_explainable_partial_dependence(
    partial_dependencies, importances, figsize, feature_names=None, model_name=None, label=0, top_n=10
):
    if feature_names is None:
        feature_names = partial_dependencies.feature_names[:top_n]
    feature_names = feature_names[:top_n]

    fr_df = fluctuation_ratio(partial_dependencies, importances, top_n=top_n, aggregated=False)
    df = importances.as_dataframe().set_index("Variable")
    df = (
        pd.concat([df, fr_df], axis=1)
        .dropna()
        .sort_values("Importance", ascending=False)
        .reset_index()
        .rename({"index": "Variable"}, axis=1)
    )

    # Set default figure size if not provided
    if figsize is None:
        figsize = (20, 5)

    ncols = 5  # Number of columns for subplots

    if top_n is None:
        top_n = min(ncols, len(feature_names))  # Use top N features or the maximum columns

    fig = plt.figure(figsize=figsize)  # Create subplots
    color_map = cm.get_cmap("Blues")  # Color map for individual curves
    if model_name is not None:
        fig.suptitle(model_name)  # Set title if model name is provided

    df = df.set_index("Variable")  # Set index to 'Variable' for easy access

    # Plot the top N features
    for i, feature_name in enumerate(feature_names):
        # Get individual values and grid values for each feature
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=label, data_type="individual")
        grid_values = partial_dependencies.get_value(feature_name=feature_name, label=label, data_type="grid_values")
        average = partial_dependencies.get_value(feature_name=feature_name, label=label, data_type="average")

        # Convert individual curves to a matrix for standard deviation calculation
        individuals_matrix = np.array(individuals)
        oscillations = get_oscillations_from_individuals(grid_values, individuals)  # Calculate oscillations
        indexes = np.argsort(oscillations)[::-1][:15]  # Select top oscillating curves
        num_curves = len(indexes)

        # Plot individual curves with varying transparency
        ax = fig.add_subplot(1, top_n, i + 1)  # Crear un subplot en una cuadrícula 2x2

        for idx, index in enumerate(indexes):
            ax.plot(grid_values, individuals[index], color=color_map(idx / num_curves), alpha=0.2)

        # Calculate standard deviation at each grid point
        std_dev = np.std(individuals_matrix, axis=0)

        # Plot the average curve
        imp = df.loc[feature_name, "Importance"]  # Get importance of the feature
        ax.set_facecolor((1, 0, 0, imp))  # Set the background color based on importance

        ax.plot(grid_values, average, color="blue", linewidth=2, label="Average")

        # Plot the confidence band (average ± standard deviation)
        ax.fill_between(grid_values, average - std_dev, average + std_dev, color="blue", alpha=0.1, label="Std Dev")

        # Adjust labels and titles
        short_feature_name = feature_name.split("_")[-1]  # Use short version of the feature name
        ax.set_xlabel("Grid Values")
        ax.set_ylabel("Partial Dependence")
        ax.set_title(f"[feature={short_feature_name}, F={df.loc[feature_name,'Fluctuation Ratio']:.3f}]", fontsize=12)
        ax.legend()

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set the number of x-axis ticks
        ax.grid(True)  # Show grid
        plt.tight_layout()  # Adjust layout to avoid overlap
