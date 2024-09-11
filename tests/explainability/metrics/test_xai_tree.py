from holisticai.datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from holisticai.explainability.metrics import tree_explainability_metrics
from holisticai.explainability.metrics import (
    weighted_average_depth,
    weighted_average_explainability_score,
    weighted_tree_gini,
    tree_depth_variance,
)
import numpy as np
import pytest

ATOL = 5e-2


@pytest.fixture
def input_data_classification():
    # classification task
    dataset = load_dataset('clinical_records', protected_attribute='sex', preprocessed=True)
    train_test = dataset.train_test_split(test_size=0.2, random_state=42)

    train = train_test['train']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(train['X'], train['y'])
    return model


@pytest.fixture
def input_data_regression():
    # regression task
    dataset = load_dataset('student', protected_attribute='sex', preprocessed=True)
    train_test = dataset.train_test_split(test_size=0.2, random_state=42)

    train = train_test['train']

    model = DecisionTreeRegressor(random_state=42)
    model.fit(train['X'], train['y'])
    return model

def test_xai_tree_classification_metrics(input_data_classification):
    model = input_data_classification
    tree = model.tree_

    metrics = tree_explainability_metrics(tree)
    assert np.isclose(
        metrics.loc["Weighted Average Depth"].value, 5.702929, atol=ATOL
    )
    assert np.isclose(
        metrics.loc["Weighted Average Explainability Score"].value, 5.142259, atol=ATOL
    )
    assert np.isclose(
        metrics.loc["Weighted Gini Index"].value, 0.911898, atol=ATOL)
    assert np.isclose(
        metrics.loc["Tree Depth Variance"].value, 3.575510, atol=ATOL
    )


def test_xai_tree_classification_metrics_separated(input_data_classification):
    model = input_data_classification
    tree = model.tree_
    
    value = weighted_average_depth(tree)
    assert np.isclose(value, 5.702929, atol=ATOL)

    value = weighted_average_explainability_score(tree)
    assert np.isclose(value, 5.142259, atol=ATOL)

    value = weighted_tree_gini(tree)
    assert np.isclose(value, 0.911898, atol=ATOL)

    value = tree_depth_variance(tree)
    assert np.isclose(value, 3.575510, atol=ATOL)

def test_xai_tree_regression_metrics(input_data_regression):
    model = input_data_regression
    tree = model.tree_

    metrics = tree_explainability_metrics(tree)
    assert np.isclose(
        metrics.loc["Weighted Average Depth"].value, 10.300633, atol=ATOL
    )
    assert np.isclose(
        metrics.loc["Weighted Average Explainability Score"].value, 9.398734, atol=ATOL
    )
    assert np.isclose(
        metrics.loc["Weighted Gini Index"].value, 0.000000, atol=ATOL)
    assert np.isclose(
        metrics.loc["Tree Depth Variance"].value, 9.422933, atol=ATOL
    )

def test_xai_tree_regression_metrics_separated(input_data_regression):
    model = input_data_regression
    tree = model.tree_

    value = weighted_average_depth(tree)
    assert np.isclose(value, 10.300633, atol=ATOL)

    value = weighted_average_explainability_score(tree)
    assert np.isclose(value, 9.398734, atol=ATOL)

    value = weighted_tree_gini(tree)
    assert np.isclose(value, 0.000000, atol=ATOL)

    value = tree_depth_variance(tree)
    assert np.isclose(value, 9.422933, atol=ATOL)
