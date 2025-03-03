"""
Module description:
-------------------
- This module contains functions to plot the dataset shift analysis results.

This module provides a collection of plot functions for 2D datasets, focusing
on displaying model predictions, neighborhood analysis, and accuracy
degradation profiles. The functions are designed to help users understand how
models behave under different scenarios, such as changes in data, prediction
performance, and neighborhood-based analysis. Through intuitive scatter plots
and additional visual aids, users can assess model predictions, compare actual
vs. predicted values, and analyze model robustness under varying conditions.

Functions included:
-------------------
- plot_2d: Generates a 2D scatter plot for a dataset, with options to highlight
  or exclusively display a subset of points. It supports both general
  visualizations and focused views of specific data points.

- plot_label_and_prediction: Creates a scatter plot that shows both actual
  labels and predicted labels for a dataset, with a vertical offset to visually
  differentiate between the two. Ideal for visual comparison of prediction
  accuracy.

- plot_neighborhood: Visualizes the neighborhoods around specified points of
  interest by plotting a convex hull, the nearest neighbors, and the accuracy
  within those neighbors. It helps users understand how local groups of data
  points contribute to overall model performance.

- plot_adp_and_adf: Plots the accuracy degradation profile (ADP) by showing the
  percentage of samples above a threshold versus the size factor of the
  dataset. It highlights key points of degradation with color-coding and
  vertical markers, providing insights into the model's robustness as data
  availability decreases. Accuracy degradation factor (ADF) is also showed
  as a circle at the first degradation point.

This module offers a set of functions to explore the behavior and performance
of machine learning models visually, facilitating a better understanding of
model predictions and robustness across different data scenarios.
"""

# Importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_and_extract_data(data):
    """
    Validates and extracts data from pandas DataFrame or NumPy arrays.

    Parameters:
    ----------
    data : pd.DataFrame or pd.Series or np.ndarray
        The data to validate and convert to a NumPy array if necessary.
        If it is a DataFrame or Series, it will be converted to a NumPy array.

    Returns:
    -------
    data_vals : np.ndarray
        The data as a NumPy array.

    Raises:
    -------
    ValueError
        If `data` is a pandas DataFrame and does not have exactly two columns.
    TypeError
        If `data` is neither a pandas DataFrame nor a NumPy array.
    """

    # Check if data is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 2:
            msg = "Data should have exactly two columns."
            raise ValueError(msg)
        data_vals = data.values  # Convert DataFrame to NumPy array
    # Check if data is a pandas Series
    elif isinstance(data, pd.Series):
        data_vals = data.to_numpy()
    # Check if data is a NumPy array
    elif isinstance(data, np.ndarray):
        data_vals = data

    # Raise error if the data is not a DataFrame or NumPy array
    else:
        msg = "Data must be either a pandas DataFrame or a NumPy array."
        raise TypeError(msg)

    return data_vals


def plot_2d(X, y, highlight_group=None, show_just_group=None, features_to_plot=None):
    """
    Plots a 2D scatter plot of a dataset, with options to highlight or exclusively
    show a subset of points.

    This function generates a 2D scatter plot from the given dataset, where each
    point represents a sample. Users can highlight specific points or display only
    a subset of the data. Axis labels can be customized using feature names from the
    `features_to_plot` argument.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The feature matrix, where each row is a sample and each column represents
        a feature. This can be either a NumPy array or a pandas DataFrame with two columns.

    y : np.ndarray or pd.Series
        The labels for each sample, represented as a one-dimensional NumPy array or
        pandas Series.

    highlight_group : list or np.ndarray, optional
        The indices of the points to be highlighted (outlined in red) or exclusively
        plotted. If None, no points are highlighted.

    show_just_group : bool, optional
        If True, only the points specified in `highlight_group` are plotted, and all
        other points are hidden. If False, all points are plotted, but the highlighted
        group is outlined.

    features_to_plot : list, optional
        A list of feature names (strings) to label the x and y axes of the plot. The
        list should contain exactly two elements. If not provided, the function will
        infer the names of `X` and `y` from the argument names.

    Returns
    -------
    None
        This function does not return any values. It displays the scatter plot.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>>
    >>> X = pd.DataFrame(
    ...     {"Feature1": np.random.rand(100), "Feature2": np.random.rand(100)}
    ... )
    >>> y = pd.Series(np.random.randint(0, 2, size=100))
    >>> highlight_group = [10, 20, 30, 40]
    >>> plot_2d(
    ...     X,
    ...     y,
    ...     highlight_group=highlight_group,
    ...     show_just_group=True,
    ...     features_to_plot=["Feature1", "Feature2"],
    ... )

    Scatter Plot of a 2D dataset:

    .. image:: /_static/images/plot_2d_pure.png
        :alt: Scatter Plot of a 2D dataset


    Scatter Plot of a 2D dataset with a highlighted group:

    .. image:: /_static/images/plot_2d_highlight_group.png
        :alt: Scatter Plot of a 2D dataset with a highlighted group


    Scatter Plot of a 2D dataset with a highlighted group and it's labels:

    .. image:: /_static/images/plot_2d_show_just_group.png
        :alt: Scatter Plot of a 2D dataset with a highlighted group and it's labels


    Scatter Plot of a 2D dataset with y_test and y_pred together in the same graph while
    caltulating the accuracy over the point and its' selected neighbors.

    .. image:: /_static/images/plot_2d_neighborhood.png
        :alt: Scatter Plot of a 2D dataset with y_test and y_pred together with neighborhood accuracy calculation
    """

    import inspect

    frame = inspect.currentframe()
    arg_info = inspect.getargvalues(frame)

    # Get the name of the variables
    if features_to_plot is not None:
        X_name = features_to_plot[0]
        y_name = features_to_plot[1]

    else:
        X_name = next(name for name, value in arg_info.locals.items() if value is X)
        y_name = next(name for name, value in arg_info.locals.items() if value is y)

    # Validate and extract data
    X_vals = _validate_and_extract_data(X)
    y_vals = _validate_and_extract_data(y)

    plt.figure(figsize=(8, 6))

    if show_just_group and highlight_group is not None:
        # Plot only the samples at highlight_group
        plt.scatter(
            X_vals[highlight_group, 0],
            X_vals[highlight_group, 1],
            c=y_vals[highlight_group],
            cmap="viridis",
            s=50,
            edgecolor="k",
        )

        # Annotate the points with their indices
        for idx in highlight_group:
            plt.text(X_vals[idx, 0], X_vals[idx, 1], str(idx), color="grey", fontsize=10, ha="right")

    else:
        # Plot all samples
        plt.scatter(X_vals[:, 0], X_vals[:, 1], c=y_vals, cmap="viridis", s=50, edgecolor="k")

        # If highlight_group is provided, outline the selected points
        if highlight_group is not None:
            if not isinstance(highlight_group, (np.ndarray, list)):
                msg = "highlight_group must be either a list or a NumPy array."
                raise TypeError(msg)

            plt.scatter(
                X_vals[highlight_group, 0],
                X_vals[highlight_group, 1],
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                s=150,
            )

    plt.xlabel(X_name, fontweight="bold")
    plt.ylabel(y_name, fontweight="bold")
    plt.title("2D Dataset Scatter Plot")


def plot_label_and_prediction(X, y, y_pred, vertical_offset=0.1, features_to_plot=None):
    """
    Plots a 2D scatter plot of a dataset, displaying both the true labels (`y`) and the
    predicted labels (`y_pred`) with a slight vertical offset for distinction.

    This function generates a scatter plot where each point represents a sample from
    the dataset. The true labels are shown in a darker shade, while the predicted labels
    are displayed with a vertical offset to distinguish them. The axes can be labeled
    using feature names provided by the user or inferred from the data.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The feature matrix where each row represents a sample and each column
        represents a feature. This can be either a NumPy array or a pandas DataFrame.

    y : np.ndarray or pd.Series
        The true labels for each sample. This can be a one-dimensional NumPy array or
        a pandas Series.

    y_pred : np.ndarray or pd.Series
        The predicted labels for each sample. This can be a one-dimensional NumPy array
        or a pandas Series.

    vertical_offset : float, optional (default=0.1)
        The vertical offset applied to the predicted labels on the plot to distinguish
        them from the true labels.

    features_to_plot : list, optional
        A list of feature names (strings) to label the x and y axes of the plot.
        If not provided, the function will infer the names of `X` and `y` from the argument names.

    Returns
    -------
    None
        This function does not return any value. It displays the scatter plot
        with true labels and predicted labels.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>>
    >>> # Example dataset
    >>> X = pd.DataFrame(
    ...     {"Feature1": np.random.rand(100), "Feature2": np.random.rand(100)}
    ... )
    >>> y = pd.Series(np.random.randint(0, 2, size=100))
    >>> y_pred = pd.Series(np.random.randint(0, 2, size=100))
    >>>
    >>> # Plot with labels and predictions
    >>> plot_label_and_prediction(
    ...     X, y, y_pred, vertical_offset=0.1, features_to_plot=["Feature1", "Feature2"]
    ... )
    This will display a 2D scatter plot with both the true labels and the predicted
    labels, where the predicted labels are slightly offset for clarity.

    Scatter Plot of a 2D dataset with y_test and y_pred together in the same graph.
    The predicted values (`y_pred`, shaded circles) are shifted vertically by a small
    amount to allow better visualization. The plot highlights areas where the classifier
    incorrectly predicted the true labels, evident by differing colors between `y_test`
    and `y_pred`.

    .. image:: /_static/images/plot_2d_label_and_prediction.png
        :alt: Scatter Plot of a 2D dataset with y_test and y_pred together in the same graph
    """

    import inspect

    frame = inspect.currentframe()
    arg_info = inspect.getargvalues(frame)

    # Get the name of the variables
    if features_to_plot is not None:
        X_name = features_to_plot[0]
        y_name = features_to_plot[1]

    else:
        X_name = next(name for name, value in arg_info.locals.items() if value is X)
        y_name = next(name for name, value in arg_info.locals.items() if value is y)

    # Validate and extract data
    X_vals = _validate_and_extract_data(X)
    y_vals = _validate_and_extract_data(y)
    y_pred_vals = _validate_and_extract_data(y_pred)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_vals[:, 0],
        X_vals[:, 1],
        c=y_vals,
        cmap="viridis",
        s=50,
        edgecolor="k",
        label="label (darker)",
    )

    # Plot y_pred with a vertical offset
    plt.scatter(
        X_vals[:, 0],
        X_vals[:, 1] - vertical_offset,
        c=y_pred_vals,
        cmap="viridis",
        s=50,
        edgecolor="k",
        label="prediction (lighter)",
        alpha=0.5,
    )

    plt.xlabel(X_name, fontweight="bold")
    plt.ylabel(y_name, fontweight="bold")
    plt.title("2D Dataset: label and prediction")
    plt.legend()


def plot_neighborhood(
    X,
    y,
    y_pred,
    n_neighbors,
    points_of_interest,
    vertical_offset=0.1,
    features_to_plot=None,
    ax=None,
    indices_show=None,
):
    """
    Plots a 2D scatter plot of the dataset, highlighting the neighborhood of specific
    points of interest and calculating accuracy over the selected neighbors.

    This function visualizes the neighborhood of selected points in a 2D dataset
    using the k-nearest neighbors algorithm. The convex hull of the points and their
    neighbors is plotted, and the accuracy of predictions within this neighborhood is
    displayed. The plot shows both true labels and predicted labels with a slight
    vertical offset for clarity.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The feature matrix where each row represents a sample and each column
        represents a feature. It can be either a NumPy array or pandas DataFrame.

    y : np.ndarray or pd.Series
        The true labels for each sample. It can be either a NumPy array or pandas Series.

    y_pred : np.ndarray or pd.Series
        The predicted labels for each sample. It can be either a NumPy array or pandas Series.

    n_neighbors : int
        The number of nearest neighbors to consider when identifying the neighborhood
        of each point of interest.

    points_of_interest : list or np.ndarray
        A list or array of indices corresponding to the points whose neighborhoods
        will be highlighted.

    vertical_offset : float, optional (default=0.1)
        The vertical offset applied to the predicted labels on the plot to distinguish
        them from the true labels.

    features_to_plot : list, optional
        A list of feature names (strings) to label the x and y axes of the plot.
        If not provided, the function will infer the names of `X` and `y` from the
        argument names.

    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes will be created.

    indices_show : list or np.ndarray, required
        The indices of the points to be shown on the plot. If a point from `points_of_interest`
        is not included in `indices_show`, a ValueError is raised.

    Returns
    -------
    None
        This function does not return any value. It displays the scatter plot with neighborhoods.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from matplotlib import pyplot as plt
    >>> X = pd.DataFrame(
    ...     {"Feature1": np.random.rand(100), "Feature2": np.random.rand(100)}
    ... )
    >>> y = pd.Series(np.random.randint(0, 2, size=100))
    >>> y_pred = pd.Series(np.random.randint(0, 2, size=100))
    >>> points_of_interest = [10, 50]
    >>> plot_neighborhood(
    ...     X=X,
    ...     y=y,
    ...     y_pred=y_pred,
    ...     n_neighbors=3,
    ...     points_of_interest=points_of_interest,
    ...     indices_show=np.arange(100),
    ...     features_to_plot=["Feature1", "Feature2"],
    ... )

    The plot will display the convex hull around the neighborhoods of the points
    of interest and annotate the accuracy of predictions over these neighbors.

    Raises
    ------
    ValueError
        If a point in `points_of_interest` is not present in `indices_show`.

    Notes
    -----
    - The convex hull of each point's neighborhood is plotted as a red dashed line.
    - The accuracy over the nearest neighbors is calculated and annotated next to
      each point of interest.
    - The function uses the k-nearest neighbors algorithm to find neighbors and
      create neighborhoods.
    """

    import inspect

    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import NearestNeighbors

    frame = inspect.currentframe()
    arg_info = inspect.getargvalues(frame)

    # Get the name of the variables
    if features_to_plot is not None:
        X_name = features_to_plot[0]
        y_name = features_to_plot[1]
    else:
        X_name = next(name for name, value in arg_info.locals.items() if value is X)
        y_name = next(name for name, value in arg_info.locals.items() if value is y)

    # Neighborhood on X
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(X)

    # Validate and extract data
    X = _validate_and_extract_data(X)
    y = _validate_and_extract_data(y)
    y_pred = _validate_and_extract_data(y_pred)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plotting the data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

    # Plot y_pred with a vertical offset
    ax.scatter(
        X[:, 0], X[:, 1] - vertical_offset, c=y_pred, cmap="viridis", s=50, edgecolor="k", label="y_pred", alpha=0.5
    )

    for sample_index in points_of_interest:
        if sample_index not in indices_show:
            msg = f"The point {sample_index} is not a point in 'indices_show'."
            raise ValueError(msg)

        # Find the position of sample_index in indices_show
        position = np.where(indices_show == sample_index)[0]

        # Find the n nearest neighbors of the sample
        _, indices = knn.kneighbors([X[position[0]]])

        # Extract the selected points (sample + neighbors)
        selected_points = X[indices[0]]

        # Create a convex hull around the selected points
        hull = ConvexHull(selected_points)

        # Plot the convex hull as an outline
        for simplex in hull.simplices:
            ax.plot(selected_points[simplex, 0], selected_points[simplex, 1], "r--", linewidth=1)

        # Annotate all points with their indices
        for i, (x_plot, y_plot) in enumerate(X):
            ax.text(x_plot, y_plot, str(indices_show[i]), color="grey", fontsize=10, ha="right")

        # Accuracy over the neighbors
        acc = accuracy_score(y[indices][0], y_pred[indices][0])

        # Add text near sample_index
        plt.text(
            X[position[0]][0],
            X[position[0]][1],
            f"Acc = {acc*100:.1f}%",
            ha="left",
            va="bottom",
            fontsize=12,
            color="blue",
        )

    # Plot labels and title
    ax.set_xlabel(X_name, fontweight="bold")
    ax.set_ylabel(y_name, fontweight="bold")

    plt.title(
        f"Convex Hull of Samples {', '.join(map(str, points_of_interest))} and its {n_neighbors} Nearest Neighbors."
    )


def plot_adp_and_adf(results_df):
    """
    Plots the Accuracy Degradation Profile (ADP) in a 2D plot, showing the percentage
    of samples above the threshold (ADP) on the vertical axis and dataset size (size_factor)
    on the horizontal axis, with the x-axis reversed.

    Points are colored green if the model's performance is acceptable ("OK") and red
    if there is significant accuracy degradation ("acc degrad!"). The first point where
    performance degradation occurs is highlighted and circled (Accuracy Degradation Factor, ADF),
    and a vertical dotted line is drawn at this point to mark the corresponding dataset size.

    Parameters
    ----------
    results_df : pd.DataFrame
        A DataFrame containing the following columns:
        - 'size_factor' (float): Fraction of the dataset used in the evaluation.
        - 'ADP' (float): The percentage of samples above the threshold for acceptable accuracy.
        - 'decision' (str): A string indicating whether the model's performance is acceptable
          ('OK') or shows significant accuracy degradation ('acc degrad!').
        - 'average_accuracy' (float): The average accuracy across the samples.
        - 'variance_accuracy' (float): The variance of the accuracy across the samples.

    Returns
    -------
    None
        This function does not return any values. It generates and displays a scatter plot
        with labeled points and a vertical line at the first accuracy degradation point.

    Example
    -------
    >>> import pandas as pd
    >>> data = {
    ...     "size_factor": [0.95, 0.9, 0.85, 0.8, 0.75],
    ...     "ADP": [0.98, 0.97, 0.94, 0.87, 0.76],
    ...     "decision": ["OK", "OK", "OK", "acc degrad!", "acc degrad!"],
    ...     "average_accuracy": [0.97, 0.96, 0.93, 0.85, 0.74],
    ...     "variance_accuracy": [0.02, 0.03, 0.04, 0.05, 0.06],
    ... }
    >>> results_df = pd.DataFrame(data)
    >>> plot_adp_and_adf(results_df)

    This will display a scatter plot with 'size_factor' on the x-axis (in reverse order)
    and 'ADP' on the y-axis. The first point where performance degrades ('acc degrad!')
    will be circled, and a vertical dotted line will be added to indicate the corresponding
    size factor.

    Notes
    -----
    - The blue line represents the average accuracy with shaded areas indicating the
      variance. Green points indicate acceptable performance, while red points mark
      instances of degradation.
    - The first 'acc degrad!' point is highlighted with a red circle and a vertical
      dotted line to emphasize the Accuracy Degradation Factor (ADF).
    - The x-axis is inverted to show the dataset size decreasing from left to right.
    """

    # Extract relevant columns
    x = results_df["size_factor"]
    y = results_df["ADP"]
    decision = results_df["decision"]
    average_accuracy = results_df["average_accuracy"]
    variance_accuracy = results_df["variance_accuracy"]

    # Create figure
    plt.figure(figsize=(10, 6))

    plt.plot(x, average_accuracy, "-o", color="blue", label="average_accuracy")
    plt.fill_between(
        x,
        average_accuracy - 0.95 * variance_accuracy,
        average_accuracy + 0.95 * variance_accuracy,
        color="blue",
        alpha=0.2,
    )

    # Plot OK points (green)
    plt.scatter(x[decision == "OK"], y[decision == "OK"], color="green", label="ADP - OK", s=100, edgecolor="k")

    # Plot acc degrad! points (red)
    plt.scatter(
        x[decision == "acc degrad!"],
        y[decision == "acc degrad!"],
        color="red",
        label="ADP - acc degrad!",
        s=100,
        edgecolor="k",
    )

    # Find the first 'acc degrad!' point
    first_degradation = results_df[results_df["decision"] == "acc degrad!"].iloc[0]

    # Highlight the first 'acc degrad!' point
    plt.scatter(
        first_degradation["size_factor"],
        first_degradation["ADP"],
        s=300,
        facecolors="none",
        edgecolors="red",
        label="ADF",
        linewidth=2,
    )

    # Add vertical dotted line at the first degradation point
    plt.axvline(x=first_degradation["size_factor"], color="red", linestyle="--")

    # Add the size factor text directly on the x-axis
    plt.text(
        first_degradation["size_factor"],
        plt.gca().get_ylim()[0] - 0.04,  # Just below the x-axis
        f'{first_degradation["size_factor"]:.2f}',
        color="red",
        ha="center",
    )

    # Label axes and title
    plt.xlabel("Size Factor", fontweight="bold")
    plt.ylabel("Percentual", fontweight="bold")
    plt.title("Accuracy Degradation Profile (ADP) and Accuracy Degradation Factor (ADF)")

    # Reverse the x-axis (from 0.95 to 0.05)
    plt.gca().invert_xaxis()

    # Show legend
    plt.legend()
    plt.grid()
