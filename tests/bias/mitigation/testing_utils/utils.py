import pandas as pd
import pytest
from sklearn import metrics


from holisticai.datasets import load_dataset, concatenate_datasets
# dictionnary of metrics
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
    dataset = load_dataset("adult") # x,y,p_attr
    dataset = dataset.rename({"x":"X"}) # X,y,p_attr
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']}) # X, y, p_attr, group_a, group_b
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE) # 0-ga | 0-gb  | 1-ga | 1-gb
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

@pytest.fixture
def regression_dataset():
    dataset = load_dataset("us_crime")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.select(range(2*SHARD_SIZE))
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    return dataset.train_test_split(test_size=0.2, random_state=0)

@pytest.fixture
def multiclass_dataset():
    dataset = load_dataset("us_crime_multiclass")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

@pytest.fixture
def recommender_dataset():
    dataset = load_dataset("lastfm")
    dataset = dataset.select(range(2*SHARD_SIZE))
    return dataset.train_test_split(test_size=0.2, stratify=dataset['p_attr'], random_state=0)

@pytest.fixture
def clustering_dataset():
    dataset = load_dataset("clinical_records")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)