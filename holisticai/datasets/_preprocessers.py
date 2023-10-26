import numpy as np
import pandas as pd

from ._dataloaders import (
    load_adult,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)
from .utils_adult_dataset import process_adult_dataset
from .utils_crime_dataset import process_crime_dataset
from .utils_lastfm_dataset import process_lastfm_dataset
from .utils_law_school_dataset import process_law_school_dataset
from .utils_student_dataset import process_student_dataset


def load_dataset(
    dataset="adult", preprocessed=True, as_array=False, **kwargs
):
    """
    Loads and preprocess tutorial datasets. Allows to return the data as numpy arrays or pandas dataframes.

    Obs.: The lastfm dataset preprocessing returns a pivot matrix, which is a numpy array. Therefore, the as_array parameter is ignored for this dataset.

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
        return process_lastfm_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
