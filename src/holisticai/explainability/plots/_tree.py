from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from holisticai.utils import Importances
from sklearn.tree._export import _MPLTreeExporter


def _color_brew(n):
    """Generate n colors using the 'viridis' colormap.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    cmap = plt.get_cmap("viridis")
    color_list = []

    for i in range(n):
        color = cmap(0.075 + 0.875 * i / n)[:3]  # Get RGB values from cmap
        if color[0] > 1 or color[1] > 1 or color[2] > 1:
            raise ValueError("Color values must be in the range [0, 1] 1")
        if color[0] < 0 or color[1] < 0 or color[2] < 0:
            raise ValueError("Color values must be in the range [0, 1] 0")
        rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        color_list.append(rgb)

    return color_list


class DTExporter(_MPLTreeExporter):
    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :]
            if tree.n_classes[0] == 1 and isinstance(node_val, Iterable) and self.colors["bounds"] is not None:
                # Unpack the float only for the regression tree case.
                # Classification tree requires an Iterable in `get_color`.
                node_val = node_val.item()
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)


def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    impurity=True,
    node_ids=False,
    precision=3,
    ax=None,
    fontsize=15,
    proportion=True,
    filled=True,
    rounded=True,
):
    exporter = DTExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
    )
    return exporter.export(decision_tree, ax=ax)


def plot_surrogate(feature_importance: Importances, ax=None, **kargs):
    """
    Plots the surrogate tree for feature importance.

    Parameters
    ----------
    feature_importance: Importances
        The feature importance object.
    ax: (matplotlib.axes.Axes, optional)
        The matplotlib axes to plot the tree on. If not provided, a new figure and axes will be created.
    kargs:
        Additional keyword arguments to be passed to the `plot_tree` function.

    Returns
    -------
    ax: matplotlib.axes.Axes

    Example
    -------
    >>> plot_surrogate(feature_importance)

    The plot should look like this:

    .. image:: /_static/images/xai_plot_surrogate.png
        :alt: Plot Surrogate

    """

    if "surrogate" not in feature_importance.extra_attrs:
        raise ValueError("Surrogate key does not exist in feature_importance.extra_attrs")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(30, 10))

    plot_tree(
        feature_importance.extra_attrs["surrogate"],
        feature_names=feature_importance.values,
        max_depth=3,
        ax=ax,
        **kargs,
    )
    description = """Classification: Color indicate majority class.\nRegression: Color indicate extremity of values."""
    ax.text(0.02, 0.92, description, fontsize=15, ha="left", transform=plt.gca().transAxes)
    return ax
