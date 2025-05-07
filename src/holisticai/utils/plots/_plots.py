"""
This module implements functions for visualizing 2D datasets and their properties in
the context of dataset shift. These functions highlight different aspects of the data,
such as test set points, predicted values, and neighborhoods around specific points of
interest.

Functions included:
-------------------
- plot_graph: Plots a 2D scatter plot of the entire dataset.
- plot_highlight_test_set: Highlights a specific subset of points in the dataset.
- plot_just_test_set: Annotates all points with their indices in the dataset.
- plot_ytest_ypred: Displays both true and predicted labels on the same plot with a
slight vertical offset.
- plot_neighborhoods: Plots the neighborhoods around specified points of interest and
calculates accuracy.
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


def plot_graph(X, y, test_indices=None):
    """
    Plots a 2D scatter plot of a dataset, with a specific subset of points outlined in red (optional).

    Parameters:
    ----------
    X : np.ndarray or pd.DataFrame
        The feature matrix where each row represents a sample and each column represents a feature.
        Can be a NumPy array or a pandas DataFrame with two columns.
    y : np.ndarray or pd.Series
        The labels for each sample. Can be a NumPy array or a pandas Series.
    test_indices : list or np.ndarray, optional
        The indices of the test samples to be highlighted.

    Returns:
    -------
    None
    """

    # Validate and extract data
    X_vals = _validate_and_extract_data(X)
    y_vals = _validate_and_extract_data(y)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vals[:, 0], X_vals[:, 1], c=y_vals, cmap="viridis", s=50, edgecolor="k")

    # If test_indices are provided, outline the selected points
    if test_indices is not None:
        if not isinstance(test_indices, (np.ndarray, list)):
            msg = "test_indices must be either a list or a NumPy array."
            raise TypeError(msg)

        plt.scatter(
            X_vals[test_indices, 0], X_vals[test_indices, 1], facecolors="none", edgecolors="red", linewidths=2, s=150
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Dataset Scatter Plot")
    plt.show()


def plot_just_test_set(X, y):
    """
    Plots a 2D scatter plot of a dataset with sample indices annotated.

    Parameters:
    ----------
    X : np.ndarray or pd.DataFrame
        The feature matrix where each row represents a sample and each column represents a feature.
        Can be a NumPy array or a pandas DataFrame with two columns.
    y : np.ndarray or pd.Series
        The labels for each sample. Can be a NumPy array or a pandas Series.

    Returns:
    -------
    None
    """

    # Validate and extract data
    X_vals = _validate_and_extract_data(X)
    y_vals = _validate_and_extract_data(y)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vals[:, 0], X_vals[:, 1], c=y_vals, cmap="viridis", s=50, edgecolor="k")

    # Annotate all points with their indices
    for i, (x_coord, y_coord) in enumerate(X_vals):
        plt.text(x_coord, y_coord, str(i), color="gray", fontsize=8, ha="right")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Dataset: Just test set")
    plt.show()


def plot_ytest_ypred(X, y, y_pred, vertical_offset=0.1):
    """
    Plots a 2D scatter plot of a dataset showing the actual labels (y) and the predicted labels (y_pred) with a vertical offset.

    Parameters:
    ----------
    X : np.ndarray
        The feature matrix where each row represents a sample and each column represents a feature.
    y : np.ndarray
        The true labels for each sample.
    y_pred : np.ndarray
        The predicted labels for each sample.

    Returns:
    -------
    None
    """

    # Validate and extract data
    X_vals = _validate_and_extract_data(X)
    y_vals = _validate_and_extract_data(y)
    y_pred_vals = _validate_and_extract_data(y_pred)

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vals[:, 0], X_vals[:, 1], c=y_vals, cmap="viridis", s=50, edgecolor="k")

    # Plot y_pred with a vertical offset
    plt.scatter(
        X_vals[:, 0],
        X_vals[:, 1] - vertical_offset,
        c=y_pred_vals,
        cmap="viridis",
        s=50,
        edgecolor="k",
        label="y_pred",
        alpha=0.5,
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Your 2D Dataset: y_true and y_pred")
    plt.show()


def plot_neighborhoods(X_test, y_test, y_pred, n_neighbors, points_of_interest, vertical_offset=0.1):
    """
    Plots the neighborhoods around specified points of interest and calculates the accuracy over the selected neighbors.

    Parameters:
    ----------
    X_test : np.ndarray
        The feature matrix for the test set.
    y_test : np.ndarray
        The true labels for the test set.
    y_pred : np.ndarray
        The predicted labels for the test set.
    n_neighbors : int
        The number of nearest neighbors to consider.
    points_of_interest : list or np.ndarray
        The indices of the points of interest in the test set.

    Returns:
    -------
    None
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import NearestNeighbors

    # Neighborhood on test set
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(X_test)

    # # Validate and extract data
    X = _validate_and_extract_data(X_test)
    y = _validate_and_extract_data(y_test)
    y_pred = _validate_and_extract_data(y_pred)

    # Loop over each point of interest
    for sample_index in points_of_interest:
        # Find the n nearest neighbors of the sample
        _, indices = knn.kneighbors([X[sample_index]])

        # Extract the selected points (sample + neighbors)
        selected_points = X[indices[0]]

        # Create a convex hull around the selected points
        hull = ConvexHull(selected_points)

        # Plotting the data
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

        # Plot y_pred with a vertical offset
        plt.scatter(
            X[:, 0], X[:, 1] - vertical_offset, c=y_pred, cmap="viridis", s=50, edgecolor="k", label="y_pred", alpha=0.5
        )

        # Plot the convex hull as an outline
        for simplex in hull.simplices:
            plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], "r--", linewidth=1)

        # Annotate all points with their indices
        for i, (x_plot, y_plot) in enumerate(X):
            plt.text(x_plot, y_plot, str(i), color="gray", fontsize=10, ha="right")

        # Accuracy over the neighbors
        acc = accuracy_score(y[indices][0], y_pred[indices][0])

        # Plot labels and title
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(
            f"Convex Hull of Sample {sample_index} and its {n_neighbors} Nearest Neighbors (acc = {acc*100:.1f}%)"
        )
        plt.show()
