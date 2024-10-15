"""
This module provides tools to evaluate and visualize the robustness of machine
learning models under conditions of dataset shift by analyzing accuracy
degradation. It includes methods for generating degradation profiles and
identifying critical points where performance significantly drops. The
functions in this module help quantify how well a model maintains its
predictive performance as the available test data size decreases, enabling
users to make informed decisions about model stability in real-world scenarios.

Functions included:
-------------------
- accuracy_degradation_factor: Identifies the first percentual of the group of
interest where accuracy suffers a significant drawback.

- accuracy_degradation_profile: Generates a detailed profile showing how a
model's accuracy degrades as the test set size is iteratively reduced, enabling
analysis of the model's stability and resilience under varying conditions.

This module is particularly useful for monitoring model performance in dynamic
environments where test data may change or reduce in size.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Constants
STEP_SIZE = 0.05
DECISION_COLUMN = "decision"


def pre_process_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    test_size: float = 0.3,
    random_state: int = 42,
):
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Check if input is a DataFrame or NumPy array
    if isinstance(X, (np.ndarray, pd.DataFrame)) and isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
        # Array of indices
        indices = np.arange(X.shape[0])

        # Split the indices
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

        # Split the data using the indices
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        else:
            X_train, X_test = X[train_indices], X[test_indices]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        else:
            y_train, y_test = y[train_indices], y[test_indices]

    else:
        raise TypeError(
            "X must be a NumPy array or pandas DataFrame, and y must be a NumPy array, pandas Series, or DataFrame."
        )

    return X_train, X_test, y_train, y_test, test_indices


def accuracy_degradation_factor(df: pd.DataFrame) -> float:
    """
    Identifies the first percentual of the group of interest where accuracy suffers
    a significant drawback.

    This function analyzes a DataFrame with decision data and detects the first
    occurrence of an accuracy degradation. It then returns the corresponding
    factor size for the degradation event.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing at least the following columns:
        - `size_factor`: A float or numeric column representing the size factor at
        different levels.
        - `decision`: A string column where decisions are labeled either as 'OK' or
        'acc degrad!'.

    Returns:
    -------
    float
        The `size_factor` associated with the first occurrence of the decision
        being 'acc degrad!'. If no such decision exists, the function will raise an
        IndexError.

    Example:
    --------
    >>> import pandas as pd
    >>> data = {'size_factor': [0.95, 0.9, 0.85, 0.8],
                'decision': ['OK', 'OK', 'acc degrad!', 'acc degrad!']}
    >>> df = pd.DataFrame(data)
    >>> accuracy_degradation_factor(df)
    0.85

    Raises:
    -------
    IndexError
        If the DataFrame does not contain any 'acc degrad!' decisions.
    """

    # Find the first row where the decision is 'acc degrad!'
    first_degradation = df[df["decision"] == "acc degrad!"].iloc[0]

    # Return the corresponding size_factor
    return float(first_degradation["size_factor"])


def accuracy_degradation_profile(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_pred: pd.Series,
    n_neighbors=None,
    neighbor_estimator=None,
    baseline_accuracy: Optional[float] = None,
    threshold_percentual: float = 0.95,
    above_percentual: float = 0.90,
    step_size: float = STEP_SIZE,
) -> pd.DataFrame:
    """
    Generates an accuracy degradation profile by iteratively reducing the size
    of the nearest neighbors considered in the test set and comparing the
    classifier's accuracy against a baseline.

    This function assesses the robustness of a model by gradually reducing the
    test set size and evaluating whether the accuracy falls below a defined
    threshold. It returns a DataFrame summarizing whether the accuracy at each
    step meets the baseline accuracy or if there is degradation.

    Parameters:
    ----------
    X_test : pd.DataFrame
        The feature matrix of the test set. Each row represents a sample, and
        each column represents a feature.
    y_test : pd.Series
        The true labels for the test set. This should be a one-dimensional
        Series or array.
    y_pred : pd.Series
        The predicted labels for the test set. This should be a one-dimensional
        Series or array.
    n_neighbors : int
        The number of neighbors to consider when computing the nearest neighbors
        model.
    baseline_accuracy : Optional[float], optional
        The baseline accuracy to compare the model's performance. If not provided,
        it will be calculated based on `y_test` and `y_pred`.
    threshold_percentual : float, optional, default=0.95
        The threshold for acceptable accuracy degradation. It represents a
        percentage of the baseline accuracy that defines the minimum acceptable
        accuracy.
    above_percentual : float, optional, default=0.90
        The proportion of samples that must exceed the accuracy threshold to avoid
        being marked as degraded.
    step_size : float, optional, default=STEP_SIZE
        The step size by which to incrementally reduce the test set size in each
        iteration. It defines the rate at which the test set is reduced.

    Returns:
    -------
    pd.DataFrame
        A styled pandas DataFrame summarizing the accuracy degradation results.
        The DataFrame contains the following columns:
        - `size_factor`: Fraction of the test set used in each step.
        - `above_threshold`: Number of samples with accuracy above the threshold.
        - `percent_degradation`: Percentage of samples exceeding the threshold.
        - `decision`: Indicates whether the accuracy at the given step meets
        the threshold ('OK') or is degraded ('acc degrad!').

    Example:
    --------
    >>> from sklearn.neighbors import NearestNeighbors
    >>> import pandas as pd
    >>> from sklearn.metrics import accuracy_score
    >>> X_test = pd.DataFrame([[1.2, 3.4], [2.2, 1.8], [1.1, 4.5], [3.2, 2.1]])
    >>> y_test = pd.Series([0, 1, 0, 1])
    >>> y_pred = pd.Series([0, 1, 0, 0])
    >>> n_neighbors = 3
    >>> degradation_profile = accuracy_degradation_profile(
    ...     X_test=X_test, y_test=y_test, y_pred=y_pred, n_neighbors=n_neighbors
    ... )
    >>> degradation_profile
    # Outputs a styled DataFrame summarizing the decisions for each test set size.

    Raises:
    -------
    ValueError
        If the lengths of `X_test`, `y_test`, or `y_pred` are inconsistent, or
        if the input parameters for `threshold_percentual`, `above_percentual`,
        or `n_neighbors` are invalid.

    Notes:
    ------
    - The function assumes that the input DataFrames or Series are appropriately
    structured and that the baseline accuracy is reasonable.
    - This method applies nearest neighbors to simulate accuracy degradation by
    reducing test set size.
    """

    # Check if the step size is too small
    if step_size < (1 / X_test.shape[0]):
        raise ValueError("'step_size' is too small (less than 1 divided by the number of samples).")

    # Validate inputs
    if isinstance(X_test, pd.DataFrame) & isinstance(y_test, pd.Series):
        # Data structures
        X_test = X_test.values
        y_test = y_test.values

    if baseline_accuracy is None:
        baseline_accuracy = accuracy_score(y_test, y_pred)

    # Validate inputs
    _validate_inputs(X_test, y_test, y_pred, baseline_accuracy, threshold_percentual, above_percentual)

    if neighbor_estimator is None:
        neighbor_estimator = NearestNeighbors(n_neighbors=n_neighbors)

    # Neighborhood on test set
    neighbor_estimator.fit(X_test)

    # Calculate accuracies for varying test set sizes
    results_df, set_size_list = _calculate_accuracies(X_test, y_test, y_pred, neighbor_estimator, step_size)

    # Summarize the results
    results_summary_df = _summarize_results(results_df, baseline_accuracy, threshold_percentual, above_percentual)

    # Apply styling
    styled_df = _styled_results(results_summary_df)

    return styled_df


def _validate_inputs(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    baseline_accuracy: float,
    threshold_percentual: float,
    above_percentual: float,
) -> None:
    """
    Validates the inputs for the accuracy degradation profile function.

    This function ensures that the input arrays have matching lengths and that
    the input parameters (baseline_accuracy, threshold_percentual, and
    above_percentual) are within their expected ranges.

    Parameters:
    ----------
    X_test : np.ndarray
        The test set features array. This array is expected to have the same
        length as `y_test` and `y_pred`.
    y_test : np.ndarray
        The true labels for the test set. Must have the same length as `X_test`
        and `y_pred`.
    y_pred : np.ndarray
        The predicted labels for the test set, corresponding to `X_test`. Must
        have the same length as `X_test` and `y_test`.
    baseline_accuracy : float
        The baseline accuracy of the model, expected to be a value between 0 and 1.
    threshold_percentual : float
        The percentage threshold to determine significant accuracy degradation.
        Must be between 0 and 1.
    above_percentual : float
        The minimum percentage of samples that must exceed the accuracy threshold
        to avoid triggering an accuracy degradation alert. Must be between 0 and 1.

    Returns:
    -------
    None
        This function does not return a value. It only raises an exception if
        validation fails.

    Raises:
    -------
    ValueError
        If any of the following conditions are met:
        - The length of `X_test`, `y_test`, and `y_pred` do not match.
        - `threshold_percentual` is not in the range (0, 1].
        - `above_percentual` is not in the range (0, 1].
        - `baseline_accuracy` is not in the range (0, 1].

    Example:
    -------
    >>> import numpy as np
    >>> _validate_inputs(
    ...     np.array([1, 2, 3]),
    ...     np.array([0, 1, 0]),
    ...     np.array([0, 1, 1]),
    ...     0.9,
    ...     0.8,
    ...     0.95,
    ... )
    No error raised, input is valid.

    >>> _validate_inputs(
    ...     np.array([1, 2, 3]), np.array([0, 1]), np.array([0, 1, 1]), 0.9, 0.8, 0.95
    ... )
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


def batched(X, batch_size=100):
    n_samples = X.shape[0]

    for i in range(0, n_samples, batch_size):
        yield np.array(X[i : i + batch_size])


def _calculate_accuracies(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    knn: Any,
    step_size: float,
) -> tuple[pd.DataFrame, list[float]]:
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
        The number of nearest neighbors to consider for each test set size.
    step_size : float
        The fraction by which to reduce the test set size at each step. Must be between 0 and 1.

    Returns:
    -------
    tuple[pd.DataFrame, list[float]]
        A tuple containing:
        - pd.DataFrame: A DataFrame with the calculated accuracies for each size factor.
                        Rows represent the number of neighbors, and columns represent the size factors.
        - list[float]: A list of size factors used in the reduction process.

    Raises:
    -------
    ValueError
        If step_size is not within the valid range (0, 1].

    Example:
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> import pandas as pd

    >>> # Generate a sample classification dataset
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )

    >>> # Calculate accuracies over decreasing test set sizes
    >>> results_df, set_size_list = _calculate_accuracies(
    ...     X_test, y_test, y_pred, n_neighbors=5, step_size=0.05
    ... )

    >>> print(results_df)
    >>> print(set_size_list)
    """

    # Validate step_size input
    if not (0 < step_size <= 1):
        raise ValueError("step_size must be between 0 and 1.")

    # Auxiliary data structures
    full_set_size = X_test.shape[0]
    no_of_steps = int(1 // step_size)
    set_size_list = [(1 - ((i + 1) * step_size)) for i in range(no_of_steps)]
    n_neighbours_list = [int(full_set_size * i) for i in set_size_list]
    results = {size_factor: [] for size_factor in set_size_list}

    # TODO: Get the neighbors for each test set size (is it possible to optimize like this?)
    test_set_neighbours = []
    for batch in batched(X_test):
        batch_indexes = knn.kneighbors(batch, n_neighbors=full_set_size, return_distance=False)
        test_set_neighbours.extend(batch_indexes)
    test_set_neighbours = np.array(test_set_neighbours)
    matches = y_test[test_set_neighbours] == y_pred[test_set_neighbours]

    # Loop over different number of neighbors
    for size_factor_index, n_neighbours in enumerate(n_neighbours_list):
        if n_neighbours <= 0:
            continue  # Skip invalid neighbor size

        # Evaluate accuracy over each test set size
        size_factor = set_size_list[size_factor_index]
        # test_set_neighbours_2 = knn.kneighbors(X_test, n_neighbors=n_neighbours, return_distance=False)
        # print(test_set_neighbours_2)

        accuracy_list = np.mean(
            matches[:, :n_neighbours], axis=1
        )  # [accuracy_score(y_pred[neighbors[:n_neighbours]], y_test[neighbors[:n_neighbours]]) for neighbors in test_set_neighbours]
        # accuracy_list = np.mean(matches[:,:n_neighbours], axis=1)

        results[size_factor] = accuracy_list

    # Organize results into a DataFrame
    results_df = pd.DataFrame.from_dict(results, orient="columns")
    results_df.index = range(1, full_set_size + 1)
    results_df.columns = [round(float(size_factor), 2) for size_factor in results_df.columns]
    results_df.index.name = "n_neighbours"
    results_df.columns.name = "size_factor"

    return results_df, set_size_list


def _summarize_results(
    results_df: pd.DataFrame, baseline_accuracy: float, threshold_percentual: float, above_percentual: float
) -> pd.DataFrame:
    """
    Summarize the accuracy results by checking against the threshold for degradation.

    This function evaluates the accuracy degradation by comparing the accuracy
    at each size factor against a calculated threshold. It determines whether
    the accuracy is acceptable or indicates degradation.

    Parameters:
    ----------
    results_df : pd.DataFrame
        A DataFrame containing accuracy values for different size factors. Each
        column represents a size factor, and each row represents the accuracy for
        that size factor.
    baseline_accuracy : float
        The baseline accuracy to which the calculated accuracy is compared.
    threshold_percentual : float
        A percentage (between 0 and 1) of the baseline accuracy that sets the
        minimum acceptable accuracy.
    above_percentual : float
        A percentage (between 0 and 1) representing the proportion of accuracies
        that need to be above the threshold for the decision to be marked as 'OK'.

    Returns:
    -------
    pd.DataFrame
        A DataFrame summarizing the degradation decisions for each size factor.
        The DataFrame contains columns:
        - 'size_factor': The fraction of the original test set used.
        - 'decision': A string indicating whether the accuracy is acceptable
        ('OK') or degraded ('acc degrad!').
        - 'above_threshold': The count of samples with accuracy above the
        threshold.
        - 'percent_degradation': The percentage of samples with accuracy above the
        threshold.

    Raises:
    -------
    ValueError
        If `threshold_percentual` or `above_percentual` are not within the range
        (0, 1].

    Example:
    -------
    >>> import pandas as pd
    >>> data = {
    ...     0.95: [0.9, 0.95, 0.97],
    ...     0.90: [0.85, 0.89, 0.92],
    ...     0.85: [0.75, 0.80, 0.85],
    ... }
    >>> results_df = pd.DataFrame(data)
    >>> baseline_accuracy = 0.95
    >>> threshold_percentual = 0.90
    >>> above_percentual = 0.95

    >>> results_summary_df = _summarize_results(
    ...     results_df,
    ...     baseline_accuracy=baseline_accuracy,
    ...     threshold_percentual=threshold_percentual,
    ...     above_percentual=above_percentual,
    ... )
    >>> print(results_summary_df)
    """

    # Calculate the threshold accuracy
    threshold = threshold_percentual * baseline_accuracy

    # Initialize an empty DataFrame to store the summary of results
    results_summary_df = pd.DataFrame(
        columns=[
            "size_factor",
            "above_threshold",
            "percent_degradation",
            "average_accuracy",
            "variance_accuracy",
            "degradate",
            "decision",
        ]
    )

    # Iterate through each size_factor in the results_df
    for size_factor in results_df.columns:
        # Count how many accuracies are above the threshold for the current
        # size factor
        above_threshold = (
            results_df[size_factor] > threshold
        ).sum()  # results_df[results_df[size_factor] > threshold].shape[0]

        # Determine whether the decision is 'OK' or indicates 'acc degrad!'
        # based on above_threshold percentage
        degradate = 0 if above_threshold / results_df.shape[0] >= above_percentual else 1
        decision = "OK" if above_threshold / results_df.shape[0] >= above_percentual else "acc degrad!"

        # Create a new row with the summary data
        new_row = pd.DataFrame(
            {
                "size_factor": [size_factor],
                "percent_degradation": [above_threshold / results_df.shape[0]],
                "above_threshold": [above_threshold],
                "average_accuracy": np.mean(results_df[size_factor]),
                "variance_accuracy": np.std(results_df[size_factor]),
                "degradate": [degradate],
                "decision": [decision],
            }
        )

        # Concatenate the new row to the summary DataFrame
        results_summary_df = pd.concat([results_summary_df, new_row], ignore_index=True)

    # Adjust the index of the results summary DataFrame to start from 1
    results_summary_df.index += 1

    return results_summary_df


def _color_cells(val: str) -> str:
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
    color = "green" if val == "OK" else "red"
    return f"color: {color}"


def _styled_results(results_summary_df: pd.DataFrame, decision_column: str = DECISION_COLUMN) -> pd.DataFrame:
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
