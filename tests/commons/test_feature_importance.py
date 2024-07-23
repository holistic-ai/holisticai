import numpy as np
from holisticai.utils import Importances

def test_compute_ranked_feature_importance():
    # Create a sample feature importance dataframe
    feature_importance = Importances(feature_names=['feature1', 'feature2', 'feature3', 'feature4'], values=np.array([0.6, 0.15, 0.15, 0.1]))

    # Test filtering with default threshold (alpha=None)
    ranked_importances = feature_importance.top_alpha()
    assert np.isclose(ranked_importances.values , np.array([0.6, 0.15])).all()
    assert ranked_importances.feature_names == ['feature1', 'feature2']

    # Test filtering with custom threshold (alpha=0.8)
    ranked_importances = feature_importance.top_alpha(alpha=0.8)
    assert np.isclose(ranked_importances.values, np.array([0.6, 0.15])).all()
    assert ranked_importances.feature_names == ['feature1', 'feature2']

    # Test filtering with custom threshold (alpha=0.5)
    ranked_importances = feature_importance.top_alpha(alpha=0.5)
    assert np.isclose(ranked_importances.values , np.array([0.6])).all()
    assert ranked_importances.feature_names == ['feature1']

    # Test filtering with custom threshold (alpha=0.2)
    ranked_importances = feature_importance.top_alpha(alpha=0.2)
    assert np.isclose(ranked_importances.values , np.array([0.6])).all()
    assert ranked_importances.feature_names == ['feature1']

def test_importances_object():
    values = np.array([0.50, 0.30, 0.20])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    ranked_feature_importance = feature_importance.top_alpha(alpha=0.8)
    print(ranked_feature_importance)