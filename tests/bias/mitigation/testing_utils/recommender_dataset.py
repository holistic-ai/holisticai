import numpy as np

from holisticai.datasets import load_last_fm


def preprocess_lastfm_dataset(df, protected_attribute, user_column, item_column):
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


def post_process_recommender(df):
    df = df.fillna(0)
    data_matrix = df.to_numpy()
    return data_matrix


def process_recommender_dataset(size="small"):
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
    df_pivot, p_attr = preprocess_lastfm_dataset(
        df,
        protected_attribute=protected_attribute,
        user_column=user_column,
        item_column=item_column,
    )
    if size == "small":
        a_index = list(np.where(p_attr == 1)[0][:10])
        b_index = list(np.where(p_attr == 0)[0][:10])

        df_pivot = df_pivot.iloc[a_index + b_index]
        p_attr = p_attr[a_index + b_index]
    data_matrix = post_process_recommender(df_pivot)
    return data_matrix, p_attr
