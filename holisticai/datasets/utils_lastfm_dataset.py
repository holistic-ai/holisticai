import numpy as np

from ._dataloaders import load_last_fm
from .dataset_processing_utils import post_process_recommender


def __preprocess_lastfm_dataset(df, protected_attribute, user_column, item_column):
    """
    Performs the pre-processing step of the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to pre-process
    protected_attribute : str
        The name of the protected attribute
    user_column : str
        The name of the user column
    item_column : str
        The name of the item column

    Returns
    -------
    df_pivot : pandas.DataFrame
        The pre-processed dataframe
    p_attr : np.ndarray
        The protected attribute
    """
    from holisticai.utils import recommender_formatter

    df["score"] = np.random.randint(1, 5, len(df))
    df[protected_attribute] = df[protected_attribute] == "m"
    df = df.drop_duplicates()
    df_pivot, p_attr = recommender_formatter(
        df,
        users_col=user_column,
        groups_col=protected_attribute,
        items_col=item_column,
        scores_col="score",
        aggfunc="mean",
    )
    return df_pivot, p_attr


def process_lastfm_dataset():
    """
    Processes the lastfm dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    data_matrix : np.ndarray
        The numerical pivot array
    p_attr : np.ndarray
        The protected attribute
    """
    bunch = load_last_fm()
    df = bunch["frame"]
    protected_attribute = "sex"
    user_column = "user"
    item_column = "artist"
    df_pivot, p_attr = __preprocess_lastfm_dataset(
        df,
        protected_attribute=protected_attribute,
        user_column=user_column,
        item_column=item_column,
    )
    data_matrix = post_process_recommender(df_pivot)
    return data_matrix, p_attr
