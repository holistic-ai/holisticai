import numpy as np
import pandas as pd


def one_hot_encode_columns(df, columns_to_encode):
    """
    Convert specified columns in the dataframe to one-hot encoding and return the column mapping and the new dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    columns_to_encode : list
        List of column names to be converted to one-hot encoding.

    Returns
    -------
    column_mapping : dict
        Dictionary where keys are original column names and values are lists of new one-hot encoded column names.
    new_df : pd.DataFrame
        The dataframe with specified columns converted to one-hot encoding.
    """
    column_mapping = {}
    new_df = df.copy()

    for column in columns_to_encode:
        # Perform one-hot encoding
        one_hot = pd.get_dummies(new_df[column], prefix=column).astype(int)

        # Update the column mapping
        column_mapping[column] = list(one_hot.columns)

        # Drop the original column and concatenate the new one-hot encoded columns
        new_df = new_df.drop(columns=[column])
        new_df = pd.concat([new_df, one_hot], axis=1)

    # replace the values in the column_mapping dictionary with the index of the new columns
    for key, value in column_mapping.items():
        column_mapping[key] = [new_df.columns.get_loc(col) for col in value]

    return column_mapping, new_df


def revert_one_hot_encoding(data, column_mapping, col_names=None):
    """
    Revert one-hot encoded columns in the dataframe or numpy array back to their original categorical representation.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The input data with one-hot encoded columns.
    column_mapping : dict
        Dictionary where keys are original column names and values are lists of one-hot encoded column indexes.

    Returns
    -------
    reverted_data : pd.DataFrame or np.ndarray
        The data with one-hot encoded columns reverted to their original categorical representation.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=col_names)

    reverted_df = data.copy()

    for original_col, one_hot_indexes in column_mapping.items():
        # Get the one-hot encoded column names from their indexes
        one_hot_cols = [reverted_df.columns[idx] for idx in one_hot_indexes]

        # Reconstruct the original categorical values
        reverted_df[original_col] = reverted_df[one_hot_cols].idxmax(axis=1).apply(lambda x: x.split("_", 1)[1])

        # Drop the one-hot encoded columns
        reverted_df = reverted_df.drop(columns=one_hot_cols)

    # Convert DataFrame back to numpy array if the original input was a numpy array
    if isinstance(data, np.ndarray):
        return reverted_df.to_numpy()

    return reverted_df
