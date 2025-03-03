"""

This module implements utils functions that prepare data for plotting in the context of dataset
shift. Still under development.

"""

from __future__ import annotations

# from typing import list, tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_to_plot(
    X: pd.DataFrame, y: np.ndarray, features_to_plot: list[str], test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input dataset into training and testing sets based on selected features.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature matrix where each row represents a sample and each column represents a feature.
    y : np.ndarray
        The target labels corresponding to the samples in X.
    features_to_plot : list[str]
        The list of features (column names) to select from X for plotting.
    test_size : float, optional (default=0.3)
        The proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before applying the split.

    Returns:
    -------
    X_train : np.ndarray
        Training set feature matrix (for selected features).
    X_test : np.ndarray
        Testing set feature matrix (for selected features).
    y_train : np.ndarray
        Training set target labels.
    y_test : np.ndarray
        Testing set target labels.

    Raises:
    -------
    KeyError
        If any of the specified `features_to_plot` are not present in `X`.
    TypeError
        If `X` is not a pandas DataFrame or `y` is not a NumPy array.
    ValueError
        If the size of `X` and `y` do not match.
    """

    # Check if X is a DataFrame and y is a NumPy array
    if not isinstance(X, pd.DataFrame):
        msg = "X must be a pandas DataFrame."
        raise TypeError(msg)
    if not isinstance(y, np.ndarray):
        msg = "y must be a NumPy array."
        raise TypeError(msg)

    # Check if all the features are present in X
    missing_features = [feature for feature in features_to_plot if feature not in X.columns]
    if missing_features:
        msg = f"The following features are missing from X: {', '.join(missing_features)}"
        raise KeyError(msg)

    # Check if the length of X and y match
    if len(X) != len(y):
        msg = "The length of X and y must match."
        raise ValueError(msg)

    # Select only the columns for the chosen features to plot
    X_selected = X[features_to_plot]

    # Generate indices for splitting
    indices = np.arange(X_selected.shape[0])

    # Split indices into train and test sets
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Use the indices to split X and y into train and test sets
    X_train, X_test = X_selected.values[train_indices], X_selected.values[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
