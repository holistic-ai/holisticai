from holisticai.datasets import load_dataset
from sklearn.linear_model import LinearRegression
from holisticai.explainability.metrics import regression_explainability_metrics
from holisticai.explainability.metrics.global_importance import alpha_score, rank_alignment, position_parity, xai_ease_score
import numpy as np
import pytest

ATOL = 5e-2
@pytest.fixture
def input_data():
    dataset = load_dataset('us_crime').sample(n=1000, random_state=42)
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']

    model = LinearRegression()
    model.fit(train['X'], train['y'])
    return model, test

def get_regression_features(model, test):
    from holisticai.utils import RegressionProxy
    from holisticai.utils.feature_importances import compute_permutation_feature_importance
    from holisticai.utils.inspection import compute_partial_dependence
    
    proxy = RegressionProxy(predict=model.predict)
    importances  = compute_permutation_feature_importance(X=test['X'], y=test['y'], proxy=proxy)
    ranked_importances = importances.top_alpha(0.8)
    partial_dependencies = compute_partial_dependence(test['X'], features=ranked_importances.feature_names, proxy=proxy)
    conditional_importances  = compute_permutation_feature_importance(X=test['X'], y=test['y'], proxy=proxy, conditional=True)
    return proxy, importances, ranked_importances, conditional_importances, partial_dependencies

def test_xai_regression_metrics(input_data):
    model, test = input_data
    proxy, importances, ranked_importances, conditional_importances, partial_dependencies = get_regression_features(model, test)

    metrics = regression_explainability_metrics(importances, partial_dependencies, conditional_importances, X=test['X'], y_pred=proxy.predict(test['X']))
    assert np.isclose(metrics.loc['Rank Alignment'].value, 0.7317350088183421, atol=ATOL)
    assert np.isclose(metrics.loc['Position Parity'].value, 0.18504188712522046, atol=ATOL)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, 1.0, atol=ATOL)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, 0.0891089108910891, atol=ATOL)

def test_xai_classification_metrics_separated(input_data):
    model, test = input_data

    proxy, importances, ranked_importances, conditional_importances, partial_dependencies = get_regression_features(model, test)
       
    value = rank_alignment(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.7317350088183421, atol=ATOL)

    value = position_parity(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.18504188712522046, atol=ATOL)
    
    value = xai_ease_score(partial_dependencies, ranked_importances)
    assert np.isclose(value, 1.0, atol=ATOL)

    value = alpha_score(importances)
    assert np.isclose(value, 0.0891089108910891, atol=ATOL)

