from holisticai.utils import LocalImportances
from holisticai.explainability.metrics import data_stability
import pandas as pd
import numpy as np

def test_local_feature_importance():
    data = pd.DataFrame.from_dict({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
        "feature3": [9, 10, 11, 12],
        "feature4": [13, 14, 15, 16]
    })
    local_importance = LocalImportances(data=data, cond= pd.Series([1, 2, 1, 2]))
    conditional_local_importance = local_importance.conditional()
    assert len(conditional_local_importance) == 2

def test_local_feature_importance_to_global():
    data = pd.DataFrame.from_dict({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
        "feature3": [9, 10, 11, 12],
        "feature4": [13, 14, 15, 16]
    })
    local_importance = LocalImportances(data=data, cond= pd.Series([1, 2, 1, 2]))
    assert len(local_importance.to_global().values)==4


def test_data_stability():
    importances = pd.DataFrame({
          "feature_1": [0.10, 0.20, 0.30],
          "feature_2": [0.10, 0.25, 0.35],
          "feature_3": [0.15, 0.20, 0.30]})
    local_importances = LocalImportances(importances)
    stability_score = data_stability(local_importances)
    assert np.isclose(stability_score, 0.9900968679601143)

def test_partial_dependence():
    from holisticai.utils import PartialDependence, Importances
    from holisticai.explainability.metrics.global_importance import xai_ease_score

    partial_dependence = [[
         {'average': [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]], 'grid_values': [[1,2,3,4,5,6,7,8,9]]},
         {'average': [[0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6]], 'grid_values': [[1,2,3,4,5,6,7,8,9]]},
    ]]
    partial_dependence = PartialDependence(values=partial_dependence)
    feature_importance = Importances(values=np.array([0.5, 0.5]), feature_names=['feature1', 'feature2'])
    score = xai_ease_score(partial_dependence, feature_importance)
    assert score==0.5