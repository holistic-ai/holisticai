import numpy as np
import pandas as pd

from ._dataloaders import (
    load_adult,
    load_heart,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)
from .utils_adult_dataset import process_adult_dataset
from .utils_crime_dataset import process_crime_dataset
from .utils_heart_dataset import process_heart_dataset
from .utils_lastfm_dataset import process_lastfm_dataset
from .utils_law_school_dataset import process_law_school_dataset
from .utils_student_dataset import process_student_dataset


def load_dataset(dataset="adult", preprocessed=True, as_array=False, **kwargs):
    """
    Loads and preprocess tutorial datasets. Allows to return the data as numpy arrays or pandas dataframes.

    For 'adult' dataset, the protected attribute is 'sex', the preprocessed array version returns a tuple with four numpy arrays containing the data,
    output variable, protected group A and protected group B. The preprocessed dataframe version returns a tuple
    with three pandas dataframes containing the data, protected group A and protected group B, the target column is 'class'.

    For 'crime' dataset, the protected attribute is 'racePctWhite', the preprocessed array version returns a tuple with four numpy arrays containing the data,
    output variable, protected group A and protected group B. The preprocessed dataframe version returns a tuple
    with three pandas dataframes containing the data, protected group A and protected group B, the target column is 'ViolentCrimesPerPop'.

    For 'student' dataset, the protected attribute is 'Mjob', the preprocessed array version returns a tuple with three numpy arrays containing the data,
    output variable and protected attribute. The preprocessed dataframe version returns a tuple
    with two pandas dataframes containing the data and the protected attribute, the target column is 'target'.

    For 'law_school' dataset, the protected attribute is 'race1', the preprocessed array version returns a tuple with four numpy arrays containing the data,
    output variable, protected group A and protected group B. The preprocessed dataframe version returns a tuple
    with three pandas dataframes containing the data, protected group A and protected group B, the target column is 'target'.

    For 'lastfm' dataset, the protected attribute is 'sex', returns a tuple with two elements, the first one is a numpy array or pandas dataframe containing the pivot matrix and
    the second one is a numpy array containing the protected attribute.

    Obs.: Contrary to other datasets, the student dataset preprocessing does not return two vectors for protected groups, but only one vector. This is because
    this dataset is more suitable for analysis where more than two elements are in the protected group.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load
    preprocessed : bool
        Whether returns the preprocessed or the raw dataset
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """
    if dataset == "adult":
        if not preprocessed:
            return load_adult(**kwargs)
        return process_adult_dataset(as_array=as_array)
    elif dataset == "crime":
        if not preprocessed:
            return load_us_crime(**kwargs)
        return process_crime_dataset(as_array)
    elif dataset == "student":
        if not preprocessed:
            return load_student(**kwargs)
        return process_student_dataset(as_array)
    elif dataset == "law_school":
        if not preprocessed:
            return load_law_school(**kwargs)
        return process_law_school_dataset(as_array)
    elif dataset == "lastfm":
        if not preprocessed:
            return load_last_fm(**kwargs)
        return process_lastfm_dataset(as_array)
    elif dataset == "heart":
        if not preprocessed:
            return load_heart(**kwargs)
        return process_heart_dataset(as_array)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
