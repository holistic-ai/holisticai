from holisticai.datasets import load_dataset
from sklearn.linear_model import LinearRegression
from holisticai.explainability.metrics import regression_explainability_metrics, regression_explainability_features
from holisticai.explainability.metrics.global_importance import alpha_importance_score, rank_alignment, position_parity, xai_ease_score
import numpy as np
import pytest

@pytest.fixture
def input_data():
    dataset = load_dataset('us_crime').sample(n=1000, random_state=42)
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']

    model = LinearRegression()
    model.fit(train['X'], train['y'])
    return model, test

def test_xai_regression_metrics(input_data):
    model, test = input_data
    metrics = regression_explainability_metrics(test['X'], test['y'], model.predict)
    assert np.isclose(metrics.loc['Rank Alignment'].value, 0.7147597001763668)
    assert np.isclose(metrics.loc['Position Parity'].value, 0.13373015873015873)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, 1.0)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, 0.0891089108910891)

def test_xai_classification_metrics_separated(input_data):
    model, test = input_data

    xai_features = regression_explainability_features(test["X"], test["y"], model.predict)
        
    value = rank_alignment(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.7147597001763668)

    value = position_parity(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.13373015873015873)
    
    value = xai_ease_score(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    assert np.isclose(value, 1.0)

    value = alpha_importance_score(xai_features.feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.0891089108910891)