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
