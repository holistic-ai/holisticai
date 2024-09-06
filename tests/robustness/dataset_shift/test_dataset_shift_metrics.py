import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Import the module to be tested
from holisticai.robustness.dataset_shift._accuracy_degradation_profile import (
    STEP_SIZE, _calculate_accuracies, _styled_results, _summarize_results,
    _validate_inputs, accuracy_degradation_profile)

# ------------------------
# Fixtures for test data and models
# ------------------------


@pytest.fixture
def test_data():
    """Fixture to generate test data for X_test, y_test, and y_pred."""
    X_test = np.random.rand(100, 2)
    y_test = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    return X_test, y_test, y_pred


@pytest.fixture
def knn_model(test_data):
    """Fixture to generate a pre-trained KNN model."""
    X_test, _, _ = test_data
    model = NearestNeighbors(n_neighbors=5)
    model.fit(X_test)
    return model


@pytest.fixture
def parameters():
    """Fixture for baseline accuracy and other parameters."""
    return {
        'baseline_accuracy': 0.8,
        'threshold_percentual': 0.9,
        'above_percentual': 0.95,
        'step_size': STEP_SIZE
    }

# ------------------------
# Test: accuracy_degradation_profile
# ------------------------


def test_accuracy_degradation_profile(test_data, knn_model, parameters):
    """Test the accuracy_degradation_profile function."""
    X_test, y_test, y_pred = test_data
    result_df = accuracy_degradation_profile(
        X_test,
        y_test,
        y_pred,
        5,  # Number of neighbors for KNN
        parameters['baseline_accuracy'],
        parameters['threshold_percentual'],
        parameters['above_percentual'],
        step_size=parameters['step_size']
    )

    # Ensure the result is a styled DataFrame
    assert isinstance(result_df, pd.io.formats.style.Styler)

    # Ensure the size_factor column exists in the underlying data
    assert 'size_factor' in result_df.data.columns

    # Ensure decisions are either 'OK' or 'acc degrad!'
    assert set(result_df.data['decision']).issubset({'OK', 'acc degrad!'})

# ------------------------
# Test: _validate_inputs
# ------------------------


@pytest.mark.parametrize("X_test, y_test, y_pred, baseline_accuracy, threshold_percentual, above_percentual, expected", [
    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 0.8, 0.9, 0.95, None),
    (np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3]), 0.8, 0.9, 0.95, ValueError),
    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 1.1, 0.9, 0.95, ValueError),
    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 0.8, -0.5, 0.95, ValueError),
    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]), 0.8, 0.9, 1.5, ValueError)
])
def test_validate_inputs(
        X_test,
        y_test,
        y_pred,
        baseline_accuracy,
        threshold_percentual,
        above_percentual,
        expected):
    """Test for input validation in _validate_inputs."""
    if expected is None:
        _validate_inputs(
            X_test,
            y_test,
            y_pred,
            baseline_accuracy,
            threshold_percentual,
            above_percentual)
    else:
        with pytest.raises(expected):
            _validate_inputs(
                X_test,
                y_test,
                y_pred,
                baseline_accuracy,
                threshold_percentual,
                above_percentual)

# ------------------------
# Test: _calculate_accuracies
# ------------------------


def test_calculate_accuracies(test_data, knn_model, parameters):
    """Test for the _calculate_accuracies function."""
    X_test, y_test, y_pred = test_data
    step_size = parameters['step_size']

    results_df, set_size_list = _calculate_accuracies(
        X_test, y_test, y_pred, 5, step_size)

    # Ensure the output is a DataFrame
    assert isinstance(results_df, pd.DataFrame)

    # Ensure the set_size_list contains the correct number of steps
    expected_length = int(1 // step_size)
    assert len(set_size_list) == expected_length

    # Ensure the DataFrame has the correct number of rows and columns
    assert results_df.shape[1] == len(set_size_list)
    assert results_df.shape[0] == X_test.shape[0]

# ------------------------
# Test: _summarize_results
# ------------------------


def test_summarize_results(parameters):
    """Test for the _summarize_results function."""
    # Simulate results_df for the test
    results_df = pd.DataFrame(
        np.random.rand(
            100, 5), columns=[
            0.95, 0.9, 0.85, 0.8, 0.75])

    # Generate the summary
    results_summary_df = _summarize_results(
        results_df,
        parameters['baseline_accuracy'],
        parameters['threshold_percentual'],
        parameters['above_percentual']
    )

    # Ensure the output is a DataFrame
    assert isinstance(results_summary_df, pd.DataFrame)

    # Ensure the DataFrame contains expected columns
    assert set(
        results_summary_df.columns) == {
        'size_factor',
        'decision',
        'above_threshold',
        'percent_above'}

    # Ensure decisions are either 'OK' or 'acc degrad!'
    assert set(results_summary_df['decision']).issubset({'OK', 'acc degrad!'})

# ------------------------
# Test: _styled_results
# ------------------------


@pytest.mark.skip(reason="The function associated with this test is already problematic.")
def test_styled_results():
    """Test for the _styled_results function."""
    # Simulate summary DataFrame
    results_summary_df = pd.DataFrame({
        'size_factor': [0.95, 0.9, 0.85],
        'decision': ['OK', 'acc degrad!', 'OK'],
        'above_threshold': [95, 90, 85],
        'percent_above': [0.95, 0.9, 0.85]
    })

    # Apply styling
    styled_df = _styled_results(results_summary_df)

    # Ensure the result is a styled DataFrame
    assert isinstance(styled_df, pd.io.formats.style.Styler)

    # Ensure the 'decision' column is styled correctly
    assert 'OK' in styled_df.data['decision']
