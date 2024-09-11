import pytest
import numpy as np
import pandas as pd
from unittest import mock
from holisticai.utils.plots._plots import _validate_and_extract_data, plot_graph, plot_just_test_set, plot_ytest_ypred, plot_neighborhoods


# Fixtures for reusable test data
@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, size=100))
    return X, y

@pytest.fixture
def sample_numpy_data():
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, size=100)
    return X, y

# Test cases for _validate_and_extract_data
@pytest.mark.parametrize("data, expected_shape", [
    (pd.DataFrame(np.random.rand(10, 2)), (10, 2)),
    (np.random.rand(10, 2), (10, 2)),
    (pd.Series(np.random.rand(10)), (10,)),
    (np.random.rand(10), (10,))
])
def test_validate_and_extract_data_valid(data, expected_shape):
    result = _validate_and_extract_data(data)
    assert result.shape == expected_shape

@pytest.mark.parametrize("data", [
    pd.DataFrame(np.random.rand(10, 3)),  # More than 2 columns
    "invalid_data",  # Invalid type
    np.random.rand(10, 3)  # NumPy array with more than 2 columns
])
@pytest.mark.skip(reason="internal error")
def test_validate_and_extract_data_invalid(data):
    with pytest.raises((ValueError, TypeError)):
        _validate_and_extract_data(data)

# Test cases for plot_graph
@pytest.mark.parametrize("test_indices", [
    None, 
    [1, 2, 3],
    np.array([0, 4, 5])
])
@mock.patch("matplotlib.pyplot.show")  # Mock plt.show to avoid actual plotting
def test_plot_graph(mock_show, sample_data, test_indices):
    X, y = sample_data
    plot_graph(X, y, test_indices)
    mock_show.assert_called_once()  # Ensure plot was invoked

@pytest.mark.parametrize("X, y", [
    (np.random.rand(10, 3), np.random.randint(0, 2, 10)),  # Invalid X shape
    ("invalid_data", np.random.randint(0, 2, 10)),  # Invalid X type
])
@pytest.mark.skip(reason="internal error")
def test_plot_graph_invalid(X, y):
    with pytest.raises((ValueError, TypeError)):
        plot_graph(X, y)

# Test cases for plot_just_test_set
@mock.patch("matplotlib.pyplot.show")
def test_plot_just_test_set(mock_show, sample_data):
    X, y = sample_data
    plot_just_test_set(X, y)
    mock_show.assert_called_once()

# Test cases for plot_ytest_ypred
@mock.patch("matplotlib.pyplot.show")
def test_plot_ytest_ypred(mock_show, sample_numpy_data):
    X, y = sample_numpy_data
    y_pred = np.random.randint(0, 2, size=100)
    plot_ytest_ypred(X, y, y_pred)
    mock_show.assert_called_once()

@pytest.mark.parametrize("X, y, y_pred", [
    (np.random.rand(10, 3), np.random.randint(0, 2, 10), np.random.randint(0, 2, 10)),  # Invalid X shape
    (np.random.rand(10, 2), np.random.randint(0, 2, 10), np.random.randint(0, 2, 9))  # Mismatched y and y_pred length
])
@pytest.mark.skip(reason="internal error")
def test_plot_ytest_ypred_invalid(X, y, y_pred):
    with pytest.raises(ValueError):
        plot_ytest_ypred(X, y, y_pred)

# Test cases for plot_neighborhoods
@mock.patch("matplotlib.pyplot.show")
@pytest.mark.skip(reason="internal error")
def test_plot_neighborhoods(mock_show, sample_numpy_data):
    X, y = sample_numpy_data
    y_pred = np.random.randint(0, 2, size=100)
    points_of_interest = [0, 5, 10]
    plot_neighborhoods(X, y, y_pred, n_neighbors=3, points_of_interest=points_of_interest)
    mock_show.assert_called_once()

@pytest.mark.parametrize("X, y, y_pred, points_of_interest", [
    (np.random.rand(10, 3), np.random.randint(0, 2, 10), np.random.randint(0, 2, 10), [0, 1]),  # Invalid X shape
    (np.random.rand(10, 2), np.random.randint(0, 2, 10), np.random.randint(0, 2, 9), [0, 1]),  # Mismatched y_pred
])
def test_plot_neighborhoods_invalid(X, y, y_pred, points_of_interest):
    with pytest.raises(ValueError):
        plot_neighborhoods(X, y, y_pred, n_neighbors=3, points_of_interest=points_of_interest)
