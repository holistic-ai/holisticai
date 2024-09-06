"""
This module implements the Accuracy Degradation Profile (ADP), a method to evaluate the robustness 
of machine learning models on datasets by iteratively reducing the test set size and analyzing the 
impact on accuracy. The module includes functions for calculating accuracy profiles, summarizing 
results, and applying color-coded decision criteria.

The ADP methodology aims on identify the degradation in model performance as the available test data 
reduces, offering insights into the resilience of the model under varying conditions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List

# Constants
STEP_SIZE = 0.05
DECISION_COLUMN = 'decision'

def accuracy_degradation_profile(
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    n_neighbors: int,
    baseline_accuracy: float,
    threshold_percentual: float = 0.95,
    above_percentual: float = 0.90,
    step_size: float = STEP_SIZE
) -> pd.DataFrame:
    """
    Generates an accuracy degradation profile by iteratively reducing the size of the nearest neighbors 
    considered in the test set and comparing the classifier's accuracy against a baseline.
    
    Parameters:
    ----------
    X_test : np.ndarray
        Test set features.
    y_test : np.ndarray
        Test set labels.
    y_pred : np.ndarray
        Predicted labels for the test set.
    n_neighbors : int
        The number of neighbors for KNN model.
    baseline_accuracy : float
        Baseline accuracy for comparison.
    threshold_percentual : float
        Threshold percentage for accuracy degradation.
    above_percentual : float
        Percentage of samples required to be above the threshold to avoid degradation.
    step_size : float, optional (default=0.05)
        Decremental step size for reducing test set size.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame summarizing the degradation decisions.
    
    Raises:
    ------
    ValueError
        If the input arrays have mismatched lengths.
    """

    # Validate inputs
    _validate_inputs(X_test, y_test, y_pred, baseline_accuracy, threshold_percentual, above_percentual)

    # Calculate accuracies for varying test set sizes
    results_df, set_size_list = _calculate_accuracies(X_test, y_test, y_pred, n_neighbors, step_size)
    
    # Summarize the results
    results_summary_df = _summarize_results(results_df, baseline_accuracy, threshold_percentual, above_percentual)
    
    # Apply styling
    styled_df = _styled_results(results_summary_df)

    return styled_df


def _validate_inputs(X_test: np.ndarray, 
                     y_test: np.ndarray, 
                     y_pred: np.ndarray, 
                     baseline_accuracy: float, 
                     threshold_percentual: float, 
                     above_percentual: float
) -> None:
    """
    Validates the inputs for the accuracy degradation profile function.

    This function ensures that the input arrays have matching lengths and that the input parameters 
    (baseline_accuracy, threshold_percentual, and above_percentual) are within their expected ranges.

    Parameters:
    ----------
    X_test : np.ndarray
        The test set features array. This array is expected to have the same length as `y_test` and `y_pred`.
    y_test : np.ndarray
        The true labels for the test set. Must have the same length as `X_test` and `y_pred`.
    y_pred : np.ndarray
        The predicted labels for the test set, corresponding to `X_test`. Must have the same length as `X_test` and `y_test`.
    baseline_accuracy : float
        The baseline accuracy of the model, expected to be a value between 0 and 1.
    threshold_percentual : float
        The percentage threshold to determine significant accuracy degradation. Must be between 0 and 1.
    above_percentual : float
        The minimum percentage of samples that must exceed the accuracy threshold to avoid triggering an accuracy degradation alert. Must be between 0 and 1.

    Raises:
    ------
    ValueError
        If any of the following conditions are met:
        - The length of `X_test`, `y_test`, and `y_pred` do not match.
        - `threshold_percentual` is not in the range (0, 1].
        - `above_percentual` is not in the range (0, 1].
        - `baseline_accuracy` is not in the range (0, 1].

    Example:
    -------
    >>> _validate_inputs(np.array([1, 2, 3]), np.array([0, 1, 0]), np.array([0, 1, 1]), 0.9, 0.8, 0.95)
    No error raised, input is valid.

    >>> _validate_inputs(np.array([1, 2, 3]), np.array([0, 1]), np.array([0, 1, 1]), 0.9, 0.8, 0.95)
    ValueError: X_test, y_test, and y_pred must have the same length.
    """

    if len(X_test) != len(y_test) or len(y_test) != len(y_pred):
        raise ValueError("X_test, y_test, and y_pred must have the same length.")
    if not (0 < threshold_percentual <= 1):
        raise ValueError("threshold_percentual must be between 0 and 1.")
    if not (0 < above_percentual <= 1):
        raise ValueError("above_percentual must be between 0 and 1.")
    if not (0 < baseline_accuracy <= 1):
        raise ValueError("baseline_accuracy must be between 0 and 1.")


def _calculate_accuracies(
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    n_neighbors: int,
    step_size: float
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Calculate accuracies by iteratively reducing the test set size and evaluating accuracy.

    This function evaluates the accuracy of predictions as the size of the test set is gradually reduced.
    For each reduction, it calculates the accuracy over a subset of the nearest neighbors in the test set.
    
    Parameters:
    ----------
    X_test : np.ndarray
        The test set features array.
    y_test : np.ndarray
        The true labels for the test set.
    y_pred : np.ndarray
        The predicted labels for the test set.
    n_neighbors : int
        The number of neighbors for KNN model.
    step_size : float
        The fraction by which to reduce the test set size at each step. Must be between 0 and 1.

    Returns:
    -------
    Tuple[pd.DataFrame, List[float]]
        A tuple containing:
        - pd.DataFrame: A DataFrame with the calculated accuracies for each size factor. The rows represent the number of neighbors, 
                        and the columns represent the size factors.
        - List[float]: A list of size factors used in the reduction process.

    Raises:
    ------
    ValueError
        If step_size is not within the valid range (0, 1].

    Example:
    -------
    >>> results_df, set_size_list = _calculate_accuracies(X_test, y_test, y_pred, n_neighbors, 0.05)
    >>> print(results_df)
    """

    # Validate step_size input
    if not (0 < step_size <= 1):
        raise ValueError("step_size must be between 0 and 1.")
    
    # Neighborhood on test set
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_test)

    # Auxiliary data structures
    full_set_size = X_test.shape[0]
    no_of_steps = int(1 // step_size)
    set_size_list = [1 - ((i + 1) * step_size) for i in range(no_of_steps)]
    n_neighbours_list = [int(full_set_size * i) for i in set_size_list]
    results = {size_factor: [] for size_factor in set_size_list}

    # Loop over different number of neighbors
    for size_factor_index, n_neighbours in enumerate(n_neighbours_list):
        if n_neighbours <= 0:
            continue  # Skip invalid neighbor size

        # Evaluate accuracy over each test set size
        size_factor = set_size_list[size_factor_index]
        test_set_neighbours = knn.kneighbors(X_test, n_neighbors=n_neighbours, return_distance=False)
        accuracy_list = [accuracy_score(y_pred[neighbors], y_test[neighbors]) for neighbors in test_set_neighbours]
        results[size_factor] = accuracy_list

    # Organize results into a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='columns')
    results_df.index = range(1, full_set_size + 1)
    results_df.columns = [round(float(size_factor), 2) for size_factor in results_df.columns]
    results_df.index.name = 'n_neighbours'
    results_df.columns.name = 'size_factor'

    return results_df, set_size_list


def _summarize_results(
    results_df: pd.DataFrame, 
    baseline_accuracy: float, 
    threshold_percentual: float, 
    above_percentual: float
) -> pd.DataFrame:
    """
    Summarize the accuracy results by checking against the threshold for degradation.

    This function evaluates the accuracy degradation by comparing the accuracy at each size factor 
    against a calculated threshold. It determines whether the accuracy is acceptable or indicates degradation.

    Parameters:
    ----------
    results_df : pd.DataFrame
        A DataFrame containing accuracy values for different size factors. 
        Each column represents a size factor, and each row represents the accuracy for that size factor.
    baseline_accuracy : float
        The baseline accuracy to which the calculated accuracy is compared.
    threshold_percentual : float
        A percentage (between 0 and 1) of the baseline accuracy that sets the minimum acceptable accuracy.
    above_percentual : float
        A percentage (between 0 and 1) representing the proportion of accuracies that need to be above 
        the threshold for the decision to be marked as 'OK'.

    Returns:
    -------
    pd.DataFrame
        A DataFrame summarizing the degradation decisions for each size factor. 
        The DataFrame contains columns:
        - 'size_factor': The fraction of the original test set used.
        - 'decision': A string indicating whether the accuracy is acceptable ('OK') or degraded ('acc degrad!').
        - 'above_threshold': The count of samples with accuracy above the threshold.
        - 'percent_above': The percentage of samples with accuracy above the threshold.

    Raises:
    ------
    ValueError
        If `threshold_percentual` or `above_percentual` are not within the range (0, 1].

    Example:
    -------
    >>> results_summary_df = _summarize_results(results_df, baseline_accuracy=0.95, threshold_percentual=0.9, above_percentual=0.95)
    >>> print(results_summary_df)
    """

    # Calculate the threshold accuracy
    threshold = threshold_percentual * baseline_accuracy

    # Initialize an empty DataFrame to store the summary of results
    results_summary_df = pd.DataFrame(columns=['size_factor', 'above_threshold', 'percent_above', 'decision'])

    # Iterate through each size_factor in the results_df
    for size_factor in results_df.columns:
        # Count how many accuracies are above the threshold for the current size factor
        above_threshold = results_df[results_df[size_factor] > threshold].shape[0]

        # Determine whether the decision is 'OK' or indicates 'acc degrad!' based on above_threshold percentage
        decision = 'OK' if above_threshold / results_df.shape[0] >= above_percentual else 'acc degrad!'

        # Create a new row with the summary data
        new_row = pd.DataFrame({
            'size_factor': [size_factor],
            'percent_above': [above_threshold / results_df.shape[0]],
            'above_threshold': [above_threshold],
            'decision': [decision]
        })

        # Concatenate the new row to the summary DataFrame
        results_summary_df = pd.concat([results_summary_df, new_row], ignore_index=True)

    # Adjust the index of the results summary DataFrame to start from 1
    results_summary_df.index += 1

    return results_summary_df


def _color_cells(val: str
) -> str:
    """
    Determines the color styling based on the value of the cell.

    Parameters:
    ----------
    val : str
        The cell value to evaluate.

    Returns:
    -------
    str
        CSS style string to apply the appropriate color.
    """
    color = 'green' if val == 'OK' else 'red'
    return f'color: {color}'


def _styled_results(results_summary_df: pd.DataFrame, 
                   decision_column: str = DECISION_COLUMN
) -> pd.DataFrame:
    """
    Apply styling to the results summary DataFrame to highlight decisions.

    Parameters:
    ----------
    results_summary_df : pd.DataFrame
        The DataFrame containing the summary results.
    decision_column : str
        The column name in the DataFrame to apply the color styling.

    Returns:
    -------
    pd.DataFrame
        DataFrame with color-coded decisions.
    """
    styled_df = results_summary_df.style.map(_color_cells, subset=[decision_column])
    return styled_df
