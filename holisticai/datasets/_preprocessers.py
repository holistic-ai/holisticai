import numpy as np
import pandas as pd

from .utils_adult_dataset import process_adult_dataset
from .utils_crime_dataset import process_crime_dataset
from .utils_lastfm_dataset import process_lastfm_dataset
from .utils_law_school_dataset import process_law_school_dataset
from .utils_student_dataset import process_student_dataset


def load_preprocessed_dataset(dataset="adult", preprocessed=False, as_array=False, **kwargs):
    """
    Loads and preprocess tutorial datasets. Allows to return the data as numpy arrays or pandas dataframes.

    Obs.: The lastfm dataset preprocessing returns a pivot matrix, which is a numpy array. Therefore, the as_array parameter is ignored for this dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """
    if dataset == "adult":
        return process_adult_dataset(as_array, preprocessed, **kwargs)
    elif dataset == "crime":
        return process_crime_dataset(as_array, preprocessed, **kwargs)
    elif dataset == "student":
        return process_student_dataset(as_array, preprocessed, **kwargs)
    elif dataset == "law_school":
        return process_law_school_dataset(as_array, preprocessed, **kwargs)
    elif dataset == "lastfm":
        return process_lastfm_dataset(preprocessed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
