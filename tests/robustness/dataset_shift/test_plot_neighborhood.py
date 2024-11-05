import pytest
import numpy as np
from unittest import mock
from matplotlib import pyplot as plt

# Import the functions to be tested
from holisticai.robustness.plots._dataset_shift import (
    plot_neighborhood,
)

@pytest.fixture
def sample_data():
    """
    Fixture to generate sample data for testing.
    Returns:
        X (np.ndarray): Feature matrix of shape (100, 2).
        y (np.ndarray): True labels of shape (100,).
        y_pred (np.ndarray): Predicted labels of shape (100,).
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    return X, y, y_pred

@pytest.fixture
def indices_show_full():
    """
    Fixture to provide full indices for display.
    Returns:
        indices_show (np.ndarray): Indices from 0 to 99.
    """
    return np.arange(100)

@pytest.mark.parametrize(
    "n_neighbors, points_of_interest, vertical_offset, features_to_plot, indices_show",
    [
        (3, [0], 0.1, None, None),
        (5, [10, 20], 0.2, ['Feature1', 'Feature2'], np.arange(100)),
        (7, [50, 60, 70], 0.05, None, np.arange(100)),
    ]
)
def test_plot_neighborhood_various_inputs(
    sample_data,
    n_neighbors,
    points_of_interest,
    vertical_offset,
    features_to_plot,
    indices_show
):
    """
    Test plot_neighborhood with various input parameters.
    """
    X, y, y_pred = sample_data
    if indices_show is None:
        indices_show = np.arange(len(X))
    else:
        # Ensure indices_show is valid
        assert np.all(np.isin(points_of_interest, indices_show)), \
            "All points_of_interest must be in indices_show."

    # Mock plt.show to prevent actual plotting during tests
    with mock.patch("matplotlib.pyplot.show"):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            vertical_offset=vertical_offset,
            features_to_plot=features_to_plot,
            indices_show=indices_show
        )
    # If no exceptions occur, test passes

def test_plot_neighborhood_with_invalid_points(sample_data, indices_show_full):
    """
    Test plot_neighborhood raises ValueError when points_of_interest not in indices_show.
    """
    X, y, y_pred = sample_data
    n_neighbors = 5
    points_of_interest = [1000]  # Invalid index
    indices_show = indices_show_full

    with pytest.raises(ValueError, match=r"The point \d+ is not a point in 'indices_show'\."):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            indices_show=indices_show
        )

def test_plot_neighborhood_with_insufficient_neighbors(sample_data, indices_show_full):
    """
    Test plot_neighborhood handles cases with insufficient neighbors gracefully.
    """
    X, y, y_pred = sample_data
    n_neighbors = 150  # More neighbors than available points
    points_of_interest = [0, 1]
    indices_show = indices_show_full

    with pytest.raises(ValueError):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            indices_show=indices_show
        )

@pytest.mark.parametrize(
    "features_to_plot",
    [
        None,
        ['Feature1', 'Feature2']
    ]
)
def test_plot_neighborhood_with_features(sample_data, features_to_plot):
    """
    Test plot_neighborhood with and without custom feature names.
    """
    X, y, y_pred = sample_data
    n_neighbors = 3
    points_of_interest = [0, 1]
    indices_show = np.arange(len(X))

    # Mock plt.show to prevent actual plotting during tests
    with mock.patch("matplotlib.pyplot.show"):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            features_to_plot=features_to_plot,
            indices_show=indices_show
        )

@pytest.mark.skip(reason="no call to this test at the moment")
def test_plot_neighborhood_plotting(sample_data, indices_show_full):
    """
    Test that plot_neighborhood calls the plotting functions correctly.
    """
    X, y, y_pred = sample_data
    n_neighbors = 3
    points_of_interest = [0, 1]
    indices_show = indices_show_full

    with mock.patch.object(plt, 'show') as mock_show, \
         mock.patch.object(plt.Axes, 'scatter') as mock_scatter, \
         mock.patch.object(plt.Axes, 'plot') as mock_plot, \
         mock.patch.object(plt.Axes, 'set_xlabel') as mock_set_xlabel, \
         mock.patch.object(plt.Axes, 'set_ylabel') as mock_set_ylabel, \
         mock.patch('matplotlib.pyplot.title') as mock_title:
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            indices_show=indices_show
        )

        # Check that plotting functions were called
        assert mock_scatter.call_count >= 1
        assert mock_plot.call_count >= 1
        mock_set_xlabel.assert_called()
        mock_set_ylabel.assert_called()
        mock_title.assert_called()
        mock_show.assert_called_once()

def test_plot_neighborhood_accuracy_calculation(sample_data, indices_show_full):
    """
    Test that accuracy is calculated correctly and text is added to the plot.
    """
    X, y, y_pred = sample_data
    n_neighbors = 3
    points_of_interest = [0]
    indices_show = indices_show_full

    with mock.patch("matplotlib.pyplot.show"), \
         mock.patch("matplotlib.pyplot.text") as mock_text:
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            indices_show=indices_show
        )

        # Check that text was added to the plot
        mock_text.assert_called()
        # Extract the accuracy value from the call arguments
        args, kwargs = mock_text.call_args
        acc_text = args[2]
        assert "Acc = " in acc_text

def test_plot_neighborhood_with_custom_axes(sample_data):
    """
    Test plot_neighborhood when a custom matplotlib axis is provided.
    """
    X, y, y_pred = sample_data
    n_neighbors = 3
    points_of_interest = [0]
    indices_show = np.arange(len(X))
    fig, ax = plt.subplots()

    with mock.patch("matplotlib.pyplot.show"):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            ax=ax,
            indices_show=indices_show
        )
        # Verify that the plot was drawn on the provided axis
        assert ax.has_data()

def test_plot_neighborhood_with_vertical_offset(sample_data):
    """
    Test plot_neighborhood with different vertical offsets.
    """
    X, y, y_pred = sample_data
    n_neighbors = 3
    points_of_interest = [0]
    indices_show = np.arange(len(X))

    vertical_offsets = [0, 0.1, 0.5]

    for offset in vertical_offsets:
        with mock.patch("matplotlib.pyplot.show"):
            plot_neighborhood(
                X,
                y,
                y_pred,
                n_neighbors,
                points_of_interest,
                vertical_offset=offset,
                indices_show=indices_show
            )
            # No exception means test passes

@pytest.mark.xfail(reason="ConvexHull from scipy.spatial presents error when the hull is flat.")
def test_plot_neighborhood_edge_cases():
    """
    Test plot_neighborhood with edge case inputs.
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1])
    n_neighbors = 3
    points_of_interest = [1]
    indices_show = np.array([0, 1])

    with mock.patch("matplotlib.pyplot.show"):
        plot_neighborhood(
            X,
            y,
            y_pred,
            n_neighbors,
            points_of_interest,
            indices_show=indices_show
        )
        # No exception means test passes
