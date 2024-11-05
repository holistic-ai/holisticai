# Base Imports
from os import environ, makedirs
from os.path import expanduser, join

import pandas as pd

# sklearn imports
from sklearn.datasets import fetch_openml


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


def load_openml(name, version, data_home=None, return_X_y=False, as_frame=True):
    """
    Generic function to load datasets from OpenML.

    Parameters
    ----------
    name : str
        Name of the dataset on OpenML.
    version : int
        Version of the dataset.
    data_home : str, optional
        The directory to which the data is downloaded. If None, we download
        to the default data home directory.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).

    Returns
    -------
    data : sklearn.utils.Bunch or tuple
        Dataset object or (data, target) tuple if return_X_y is True.
    """
    if data_home is None:
        data_home = get_data_home()

    return fetch_openml(
        name=name,
        version=version,
        data_home=data_home,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def load_student(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("UCI-student-performance-mat", 1, data_home, return_X_y, as_frame)


def load_adult(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("adult", 2, data_home, return_X_y, as_frame)


def load_law_school(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("law-school-admission-bianry", 1, data_home, return_X_y, as_frame)


def load_last_fm(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("LastFM_dataset", 1, data_home, return_X_y, as_frame)


def load_us_crime(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("us_crime", 1, data_home, return_X_y, as_frame)


def load_heart(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("heart-failure", 1, data_home, return_X_y, as_frame)


def load_german_credit(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("German-Credit-Risk-with-Target", 1, data_home, return_X_y, as_frame)


def load_census_kdd(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("Census-Income-KDD", 3, data_home, return_X_y, as_frame)


def load_bank_marketing(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("bank-marketing", 9, data_home, return_X_y, as_frame)


def load_compas_two_year_recid():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    df = data.loc[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
        & (data["is_recid"] != -1)
        & (data["c_charge_degree"] != "O")
        & (data["score_text"] != "N/A"),
        [
            "age",
            "c_charge_degree",
            "race",
            "age_cat",
            "score_text",
            "sex",
            "priors_count",
            "days_b_screening_arrest",
            "decile_score",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "v_type_of_assessment",
            "c_days_from_compas",
            "v_score_text",
            "v_decile_score",
            "two_year_recid",
        ],
    ]
    return df


def load_compas_is_recid():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    df = data.loc[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
        & (data["is_recid"] != -1)
        & (data["c_charge_degree"] != "O")
        & (data["score_text"] != "N/A"),
        [
            "age",
            "c_charge_degree",
            "race",
            "age_cat",
            "score_text",
            "sex",
            "priors_count",
            "days_b_screening_arrest",
            "decile_score",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "v_type_of_assessment",
            "c_days_from_compas",
            "v_score_text",
            "v_decile_score",
            "is_recid",
        ],
    ]
    return df


def load_diabetes(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("Diabetes-130-Hospitals_(Fairlearn)", 2, data_home, return_X_y, as_frame)


def load_acsincome(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("ACSIncome", 3, data_home, return_X_y, as_frame)


def load_acspublic(data_home=None, return_X_y=False, as_frame=True):
    return load_openml("ACSPublicCoverage", 2, data_home, return_X_y, as_frame)
