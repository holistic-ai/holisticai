import pandas as pd


def check_valid_output_type(y_true: pd.Series):
    if y_true.dtype.kind in ["i", "u", "O"]:
        return "classification"
    if y_true.dtype.kind in ["f"]:
        return "regression"
    raise ValueError(f"The target variable must be either continuous or categorical, found: {y_true.dtype}")
