# Base Imports
from os import environ, makedirs
from os.path import expanduser, join

import pandas as pd


def get_data_home(data_home=None):
    """
    Return the path of the holisticai data directory.
    By default the data directory is set to a folder named 'holisticai_data' in the
    user home folder.
    Alternatively, it can be set by the 'HOLISTICAI_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str, default=None
        The path to holisticai data directory. If `None`, the default path
        is `~/holisticai_data`.
    Returns
    -------
    data_home: str
        The path to holisticai data directory.
    """
    if data_home is None:
        data_home = environ.get("HOLISTIC_AI_DATA", join("~", "holisticai_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def load_hai_datasets(dataset_name, data_home=None):
    """
    Generic function to load datasets from holisticai datasets repository.

    Available datasets:
    - adult
    - law_school
    - student
    - lastfm
    - us_crime
    - clinical_records
    - acsincome
    - acspublic
    - compass_two_year_recid
    - compass_is_recid
    - german_credit
    - bank_marketing
    - census_kdd
    - diabetes
    - mw_small
    - mw_medium

    Parameters
    ----------
    name : str
        Name of the dataset.
    version : int
        Version of the dataset.
    data_home : str, optional
        The directory to which the data is downloaded. If None, we download
        to the default data home directory.
    Returns
    -------
    data : pd.DataFrame
        The dataset from holisticai datasets.

    References
    ----------
    .. [1] https://huggingface.co/datasets/holistic-ai/holisticai-datasets

    """
    if data_home is None:
        data_home = get_data_home()

    return pd.read_parquet(
        f"https://huggingface.co/datasets/holistic-ai/holisticai-datasets/resolve/main/data/{dataset_name}/{dataset_name}_dataset.parquet"
    )

