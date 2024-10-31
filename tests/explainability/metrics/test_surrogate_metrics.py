import pandas as pd 
from holisticai.explainability.metrics.surrogate import surrogate_features_stability, surrogate_feature_importances_stability
from holisticai.explainability.metrics.global_feature_importance import SpreadDivergence, spread_divergence
import pytest
import numpy as np
from sklearn.datasets import make_classification
from holisticai.utils.surrogate_models import MultiClassificationSurrogate

@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def surrogate(data):
    X, y = data
    return MultiClassificationSurrogate(X, y, model_type="shallow_tree")
   

def test_feature_importances_spread():
    # Test with uniform feature importances
    feature_importances = np.array([0.25, 0.25, 0.25, 0.25])
    result = spread_divergence(feature_importances)
    assert np.isclose(result, 0.0), f"Expected 1.0, got {result}"

    # Test with non-uniform feature importances
    feature_importances = np.array([0.7, 0.1, 0.1, 0.1])
    result = spread_divergence(feature_importances)
    assert 0 <= result < 1.0, f"Expected value between 0 and 1, got {result}"

    # Test with zero feature importances
    feature_importances = np.array([0.0, 0.0, 0.0, 0.0])
    result = spread_divergence(feature_importances)
    assert result == 1.0, f"Expected 0.0, got {result}"

    # Test with single feature importance
    feature_importances = np.array([1.0])
    result = spread_divergence(feature_importances)
    assert result == 1.0, f"Expected 1.0, got {result}"


def test_surrogate_feature_importances_stability(data, surrogate):
    X, y = data
    y_pred = surrogate.predict(X)
    stability_score = surrogate_feature_importances_stability(X, y_pred, surrogate, num_bootstraps=5)
    assert 0 <= stability_score <= 1, "Stability score should be between 0 and 1"

def test_surrogate_features_stability(data, surrogate):
    X, y = data
    y_pred = surrogate.predict(X)
    stability_score = surrogate_features_stability(X, y_pred, surrogate, num_bootstraps=5)
    assert 0 <= stability_score <= 1, "Stability score should be between 0 and 1"


def test_feature_importance_spread():
    # Create a simple dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 3, 4])

    surrogate = MultiClassificationSurrogate(X, y, model_type="shallow_tree")

    # Initialize the metric
    metric = SpreadDivergence()

    # Calculate the metric
    result = metric(surrogate.feature_importances_)

    # Assert the result is within expected range
    assert 0 <= result <= 1, "FeatureImportanceSpread result should be between 0 and 1"


def test_categorical_explainability_metrics(data, surrogate):
    from holisticai.explainability.metrics import classification_surrogate_explainability_metrics
    X, y = data
    y_pred = surrogate.predict(X)
    metrics = classification_surrogate_explainability_metrics(X, y, y_pred, surrogate)
    values = metrics.loc[:,'Value'] # type: ignore
    assert not values.isnull().any(), "Values should not be null"

def test_regression_explainability_metrics(data, surrogate):
    from holisticai.explainability.metrics import regression_surrogate_explainability_metrics
    X, y = data
    y_pred = surrogate.predict(X)
    metrics = regression_surrogate_explainability_metrics(X, y, y_pred, surrogate)
    values = metrics.loc[:,'Value'] # type: ignore
    assert not values.isnull().any(), "Values should not be null"

def test_clustering_explainability_metrics(data):
    from holisticai.explainability.metrics import clustering_surrogate_explainability_metrics
    X, y = data
    metrics = clustering_surrogate_explainability_metrics(X, y, surrogate_type="shallow_tree", metric_type="all")
    values = metrics.loc[:,'Value'] # type: ignore
    assert not values.isnull().any(), "Values should not be null"