import numpy as np
import pandas as pd
from holisticai.explainability.metrics.global_feature_importance import fluctuation_ratio
from holisticai.utils import PartialDependence, Importances

def test_fluctuation_ratio_unweighted_aggregated():
    partial_dependence = PartialDependence(values=[[
        {'average': [[0.1, 0.2, 0.3]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]},
        {'average': [[0.4, 0.5, 0.6]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]]}
    ]])
    result = fluctuation_ratio(partial_dependence, aggregated=True, weighted=False)
    assert isinstance(result, float)

def test_fluctuation_ratio_weighted_aggregated():
    partial_dependence = PartialDependence(values=[[
        {'average': [[0.1, 0.2, 0.3]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]},
        {'average': [[0.4, 0.5, 0.6]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]]}
    ]], feature_names=['feature1', 'feature2'])
    importances = Importances(values=np.array([0.6, 0.4]), feature_names=['feature1', 'feature2'])
    result = fluctuation_ratio(partial_dependence, importances=importances, aggregated=True, weighted=True)
    assert isinstance(result, float)

def test_fluctuation_ratio_unweighted_non_aggregated():
    partial_dependence = PartialDependence(values=[[
        {'average': [[0.1, 0.2, 0.3]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]},
        {'average': [[0.4, 0.5, 0.6]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]]}
    ]])
    result = fluctuation_ratio(partial_dependence, aggregated=False, weighted=False)
    assert isinstance(result, pd.DataFrame)
    assert 'Fluctuation Ratio' in result.columns

def test_fluctuation_ratio_weighted_non_aggregated():
    partial_dependence = PartialDependence(values=[[
        {'average': [[0.1, 0.2, 0.3]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]},
        {'average': [[0.4, 0.5, 0.6]], 'grid_values': [[1, 2, 3]], 'individual': [[[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]]}
    ]])
    importances = Importances(values=np.array([0.6, 0.4]), feature_names=['feature1', 'feature2'])
    result = fluctuation_ratio(partial_dependence, importances=importances, aggregated=False, weighted=True)
    assert isinstance(result, pd.DataFrame)
    assert 'Fluctuation Ratio' in result.columns