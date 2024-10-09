import pytest
import numpy as np
import pandas as pd
from unittest import mock
from holisticai.robustness.metrics.dataset_shift._accuracy_degradation_profile import (
    accuracy_degradation_factor,
    accuracy_degradation_profile
)

# Mock Data
n_samples = 10

@pytest.fixture
def mock_data():
    """
    Fixture providing mock X_test, y_test, and y_pred data for all tests.
    """
    X_test = np.random.rand(n_samples, 2)  # 10 samples, 2 features
    y_test = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])  # True labels, 10 samples
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])  # Predicted labels, 10 samples
    return X_test, y_test, y_pred


@pytest.mark.parametrize(
    "size_factors, decisions, expected",
    [
        ([0.95, 0.9, 0.85, 0.8], ['OK', 'OK', 'acc degrad!', 'acc degrad!'], 0.85),
        ([0.95, 0.9, 0.85], ['OK', 'OK', 'OK'], None),
    ]
)
def test_accuracy_degradation_factor(size_factors, decisions, expected):
    """
    Parametrized test for accuracy_degradation_factor function with different decision scenarios.
    """
    decision_df = pd.DataFrame({
        'size_factor': size_factors,
        'decision': decisions
    })

    if expected is not None:
        result = accuracy_degradation_factor(decision_df)
        assert result == expected, f"Expected {expected}, but got {result}"
    else:
        # If no 'acc degrad!' found, it should raise IndexError
        with pytest.raises(IndexError):
            accuracy_degradation_factor(decision_df)


@pytest.mark.parametrize(
    "n_neighbors, baseline_acc, mock_acc_score, expected_baseline",
    [
        (3, None, 0.9, 0.9),  # Case 1: No baseline accuracy provided, accuracy_score mocked to 0.9
        (3, 0.8, 0.9, 0.8),   # Case 2: Baseline accuracy provided
    ]
)
def test_accuracy_degradation_profile(mock_data, n_neighbors, baseline_acc, mock_acc_score, expected_baseline):
    """
    Parametrized test for accuracy_degradation_profile function.
    """
    X_test, y_test, y_pred = mock_data

    # Mock the result of _calculate_accuracies and accuracy_score
    mock_calculate_accuracies = (pd.DataFrame({
        0.95: [0.9, 0.9, 0.9],
        0.9: [0.85, 0.85, 0.85],
        0.85: [0.80, 0.80, 0.80],
    }), [0.95, 0.9, 0.85])

    with mock.patch('holisticai.robustness.metrics.dataset_shift._accuracy_degradation_profile.accuracy_score', return_value=mock_acc_score), \
         mock.patch('holisticai.robustness.metrics.dataset_shift._accuracy_degradation_profile.NearestNeighbors'), \
         mock.patch('holisticai.robustness.metrics.dataset_shift._accuracy_degradation_profile._calculate_accuracies', return_value=mock_calculate_accuracies):
        
        # Run the accuracy_degradation_profile function
        result = accuracy_degradation_profile(
            pd.DataFrame(X_test), 
            pd.Series(y_test), 
            pd.Series(y_pred), 
            n_neighbors=n_neighbors, 
            baseline_accuracy=baseline_acc,
            step_size = 0.10,
        )
        
        # Assertions
        assert isinstance(result.data, pd.DataFrame), "The output should be a DataFrame"
        assert expected_baseline == mock_acc_score or expected_baseline == baseline_acc, \
            f"Expected baseline {expected_baseline}, but got {baseline_acc}"
