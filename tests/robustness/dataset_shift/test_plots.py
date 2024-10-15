import pytest
import numpy as np
import pandas as pd
from unittest import mock
from matplotlib import pyplot as plt

# Import the functions to be tested
from holisticai.robustness.plots._dataset_shift import (
    _validate_and_extract_data,
    plot_2d,
    plot_label_and_prediction,
    plot_neighborhood,
    plot_adp_and_adf,
)

@pytest.fixture
def mock_data():
    """Fixture providing mock data for tests."""
    X = np.random.rand(10, 2)  # 10 samples, indices 0-9
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])  # True labels
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])  # Predicted labels
    return X, y, y_pred

@pytest.fixture
def mock_results_df():
    """Fixture providing a mock results DataFrame."""
    return pd.DataFrame({
        'size_factor': [0.95, 0.9, 0.85, 0.8],
        'percent_degradation': [0.9, 0.85, 0.8, 0.75],
        'decision': ['OK', 'OK', 'acc degrad!', 'acc degrad!'],
        'average_accuracy': [0.95, 0.9, 0.85, 0.8],
        'variance_accuracy': [0.01, 0.02, 0.03, 0.04]
    })

def test_validate_and_extract_data():
    """Test for _validate_and_extract_data function."""
    df = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })
    series = pd.Series(np.random.rand(10))

    assert isinstance(_validate_and_extract_data(df), np.ndarray)
    assert isinstance(_validate_and_extract_data(series), np.ndarray)

    with pytest.raises(ValueError):
        _validate_and_extract_data(pd.DataFrame(np.random.rand(10, 3)))

    with pytest.raises(TypeError):
        _validate_and_extract_data("Not a DataFrame or Array")

@pytest.mark.parametrize(
    "highlight_group, show_just_group, expected_call_count",
    [
        (None, None, 0),
        ([1, 2, 3], None, 0),
        ([1, 2, 3], True, 0),
    ]
)
def test_plot_2d(mock_data, highlight_group, show_just_group, expected_call_count):
    """Test for plot_2d function."""
    X, y, _ = mock_data
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        plot_2d(X, y, highlight_group=highlight_group, show_just_group=show_just_group)
        assert mock_show.call_count == expected_call_count

@pytest.mark.parametrize(
    "vertical_offset, expected_call_count",
    [(0.1, 0), (0.2, 0)]
)
def test_plot_label_and_prediction(mock_data, vertical_offset, expected_call_count):
    """Test for plot_label_and_prediction function."""
    X, y, y_pred = mock_data
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        plot_label_and_prediction(X, y, y_pred, vertical_offset=vertical_offset)
        assert mock_show.call_count == expected_call_count

def test_plot_adp_and_adf(mock_results_df):
    """Test for plot_adp_and_adf function."""
    results_df = mock_results_df
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        plot_adp_and_adf(results_df)
        assert mock_show.call_count == 0

    # Check if the first 'acc degrad!' point is correctly identified
    first_degradation = results_df[results_df['decision'] == 'acc degrad!'].iloc[0]
    assert first_degradation['size_factor'] == 0.85

@pytest.mark.parametrize(
    "highlight_group, show_just_group, features_to_plot",
    [
        ([0, 1], False, ['X1', 'X2']),
        ([2, 3], True, None)
    ]
)
def test_plot_2d_with_features(mock_data, highlight_group, show_just_group, features_to_plot):
    """Test for plot_2d with features_to_plot."""
    X, y, _ = mock_data
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        plot_2d(
            X,
            y,
            highlight_group=highlight_group,
            show_just_group=show_just_group,
            features_to_plot=features_to_plot
        )
        assert mock_show.call_count == 0
