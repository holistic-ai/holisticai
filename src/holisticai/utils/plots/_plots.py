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


def plot_graph(X, y):
    """
    Plots a 2D scatter plot of a dataset.

    Parameters:
    ----------
    X : np.ndarray
        The feature matrix where each row represents a sample and each column represents a feature.
    y : np.ndarray
        The labels for each sample.

    Returns:
    -------
    None
    """
    import matplotlib.pyplot as plt

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Your 2D Dataset")
    plt.show()


def plot_highlight_test_set(X, y, test_indices):
    """
    Plots a 2D scatter plot of a dataset with a specific subset of points outlined in red.

    Parameters:
    ----------
    X : np.ndarray
        The feature matrix where each row represents a sample and each column represents a feature.
    y : np.ndarray
        The labels for each sample.
    test_indices : list or np.ndarray
        The indices of the test samples to be highlighted.

    Returns:
    -------
    None
    """
    import matplotlib.pyplot as plt

    # Define the indices of the points you want to outline
    selected_indices = test_indices

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

    # Outline the selected points
    plt.scatter(
        X[selected_indices, 0], X[selected_indices, 1], facecolors="none", edgecolors="red", linewidths=2, s=150
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Your 2D Dataset: Highlight test set")
    plt.show()


def plot_just_test_set(X, y):
    """
    Plots a 2D scatter plot of a dataset with sample indices annotated.

    Parameters:
    ----------
    X : np.ndarray
        The feature matrix where each row represents a sample and each column represents a feature.
    y : np.ndarray
        The labels for each sample.

    Returns:
    -------
    None
    """
    import matplotlib.pyplot as plt

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

    # Annotate all points with their indices
    for i, (x, y) in enumerate(X):
        plt.text(x, y, str(i), color="gray", fontsize=8, ha="right")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Your 2D Dataset: Just test set")
    plt.show()


def plot_ytest_ypred(X, y, y_pred):
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
    import matplotlib.pyplot as plt

    # Adjustable vertical offset
    vertical_offset = 0.1

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, edgecolor="k")

    # Plot y_pred with a vertical offset
    plt.scatter(
        X[:, 0], X[:, 1] - vertical_offset, c=y_pred, cmap="viridis", s=50, edgecolor="k", label="y_pred", alpha=0.5
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Your 2D Dataset: y_true and y_pred")
    plt.show()


def plot_neighborhoods(X_test, y_test, y_pred, n_neighbors, points_of_interest):
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

    # Adjustable vertical offset
    vertical_offset = 0.1

    # Neighborhood on test set
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_test)

    # Loop over each point of interest
    for sample_index in points_of_interest:
        # Data to plot
        X, y = X_test, y_test

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
        for i, (x, y) in enumerate(X):
            plt.text(x, y, str(i), color="gray", fontsize=10, ha="right")

        # Accuracy over the neighbors
        acc = accuracy_score(y_test[indices][0], y_pred[indices][0])

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(
            f"Convex Hull of {n_neighbors} points: Sample {sample_index} and its Nearest Neighbors (accuracy = {acc*100:.1f}%)"
        )
        plt.show()
