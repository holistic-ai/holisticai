import numpy as np
import pandas as pd
import pytest
from holisticai.xai.metrics.global_importance._xai_ease_score import XAIEaseScore, compare_tangents


@pytest.fixture
def xai_ease_score():
    return XAIEaseScore()

def test_compute_xai_ease_score_data(xai_ease_score):
    partial_dependence = {
        'feature1': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        'feature2': [0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6],
    }
    score_data = xai_ease_score.compute_xai_ease_score_data(partial_dependence)
    assert len(score_data) == 2
    assert 'feature' in score_data.columns
    assert 'scores' in score_data.columns

def test_compute_xai_ease_score(xai_ease_score):
    score_data = pd.DataFrame({'feature': ['feature1', 'feature2'], 'scores': ['Easy', 'Medium']})
    assert xai_ease_score.compute_xai_ease_score(score_data) == 0.75

def test_xai_feature_ease_score(xai_ease_score):
    partial_dependence = {
        'feature1': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        'feature2': [0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6],
    }
    assert xai_ease_score.xai_feature_ease_score(partial_dependence) == 0.5

def test_xai_ease_score(xai_ease_score):
    from holisticai.xai.commons._definitions import PartialDependence
    from holisticai.xai.commons import PermutationFeatureImportance
    partial_dependence = [
        {'average': [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]], 'grid_values': [[1,2,3,4,5,6,7,8,9]]},
        {'average': [[0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6]], 'grid_values': [[1,2,3,4,5,6,7,8,9]]},
    ]
    partial_dependence = PartialDependence(partial_dependence=partial_dependence)
    feature_importance = PermutationFeatureImportance(feature_importances=pd.DataFrame({'Variable': ['feature1', 'feature2'], 'Importance': [0.5, 0.5]}))
    assert xai_ease_score(partial_dependence, feature_importance) == 0.5
    

def test_compare_tangents():
    points = [0.1, 0.2, 0.3, 0.8, 0.10, 0.2, 0.5, 0.4]
    slopes, similarities = compare_tangents(points)
    assert np.isclose(slopes[0], -1.0)
    assert np.isclose(slopes[1], -1.0)


def test_compare_tangents_equal():
    points = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    slopes, similarities = compare_tangents(points)
    assert np.isclose(slopes[0], 1.0)
    assert np.isclose(slopes[1], 1.0)
 