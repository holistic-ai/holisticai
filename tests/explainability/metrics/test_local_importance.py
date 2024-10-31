
import numpy as np
import pandas as pd
from holisticai.explainability.metrics.local_feature_importance import feature_stability
from holisticai.explainability.metrics.local_feature_importance import rank_consistency

def test_rank_consistency_unweighted_aggregated():
    local_importances_values = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])
    result = rank_consistency(local_importances_values, weighted=False, aggregate=True)
    assert isinstance(result, float)

def test_rank_consistency_weighted_aggregated():
    local_importances_values = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])
    result = rank_consistency(local_importances_values, weighted=True, aggregate=True)
    assert isinstance(result, float)

def test_rank_consistency_unweighted_non_aggregated():
    local_importances_values = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])
    result = rank_consistency(local_importances_values, weighted=False, aggregate=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (local_importances_values.shape[1],)

def test_rank_consistency_weighted_non_aggregated():
    local_importances_values = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])
    result = rank_consistency(local_importances_values, weighted=True, aggregate=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (local_importances_values.shape[1],)

class MockLocalImportances:
    def __init__(self, data):
        self.data = {"DataFrame": pd.DataFrame(data)}

def test_feature_stability_variance():
    local_importances = MockLocalImportances(data={
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [0.4, 0.3, 0.2, 0.1]
    })
    result = feature_stability(local_importances, strategy="variance", k=2, num_samples=100)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_feature_stability_entropy():
    local_importances = MockLocalImportances(data={
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [0.4, 0.3, 0.2, 0.1]
    })
    result = feature_stability(local_importances, strategy="entropy", k=2, num_samples=100)
    assert isinstance(result, float)
    assert 0 <= result <= 1