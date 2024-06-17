import pandas as pd
import pytest
from sklearn import metrics


from holisticai.datasets import load_dataset

metrics_dict = {
    "Accuracy": metrics.accuracy_score,
    "Balanced accuracy": metrics.balanced_accuracy_score,
    "Precision": metrics.precision_score,
    "Recall": metrics.recall_score,
    "F1-Score": metrics.f1_score,
}

SHARD_SIZE=50

@pytest.fixture
def categorical_dataset():
    dataset = load_dataset("adult", protected_attribute='race') # x,y,group_a,group_b
    dataset = dataset.groupby(['y','group_a']).sample(SHARD_SIZE) # 0-ga | 0-gb  | 1-ga | 1-gb
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

@pytest.fixture
def regression_dataset():
    dataset = load_dataset("us_crime")
    dataset = dataset.select(range(2*SHARD_SIZE))
    return dataset.train_test_split(test_size=0.2, random_state=0)

@pytest.fixture
def multiclass_dataset():
    dataset = load_dataset("us_crime_multiclass")
    dataset = dataset.groupby(['y','group_a']).sample(SHARD_SIZE)
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

@pytest.fixture
def recommender_dataset():
    dataset = load_dataset("lastfm")
    dataset = dataset.select(range(2*SHARD_SIZE))
    return dataset.train_test_split(test_size=0.2, stratify=dataset['p_attr'], random_state=0)

@pytest.fixture
def clustering_dataset():
    dataset = load_dataset("clinical_records")
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

def check_results(df1, df2, atol=1e-5):
    """
    Checks if the results are equal

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe
    df2 : pandas.DataFrame
        The second dataframe

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the results
    """
    import numpy as np
    assert np.isclose(df1["Value"].iloc[0], df2["Value"].iloc[0], atol=atol), (df1["Value"].iloc[0], df2["Value"].iloc[0])
