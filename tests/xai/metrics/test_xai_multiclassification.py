from sklearn.naive_bayes import GaussianNB
from holisticai.datasets import load_dataset   
from holisticai.xai.metrics import multiclass_xai_metrics, multiclass_xai_features
from holisticai.xai.metrics.global_importance import alpha_importance_score, rank_alignment, position_parity, xai_ease_score
import numpy as np
import pytest

@pytest.fixture
def input_data():
    dataset = load_dataset('student_multiclass').sample(n=1000, random_state=42)
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']

    model = GaussianNB()
    model.fit(train['X'], train['y'])
    return model, test

@pytest.mark.parametrize("strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score", [
    ("permutation", 0.4780687830687831, 0.137010582010582, 0.85, 0.38461538461538464),
    ("surrogate",  0.2962962962962963, 0.296296296296296, 0.8333333333333333, 0.11538461538461539)
])
def test_xai_multiclassification_metrics(strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score, input_data):
    model, test = input_data
    
    metrics = multiclass_xai_metrics(test['X'], test['y'], model.predict, model.predict_proba, classes=[0,1,2], strategy=strategy)
    assert np.isclose(metrics.loc['Rank Alignment'].value, rank_alignment)
    assert np.isclose(metrics.loc['Position Parity'].value, position_parity)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, xai_ease_score)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, alpha_imp_score)


def test_xai_classification_metrics_separated(input_data):
    model, test = input_data
    
    xai_features = multiclass_xai_features(test["X"], test["y"], model.predict, model.predict_proba, classes=[0,1])
        
    value = rank_alignment(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.4780687830687831)

    value = position_parity(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.137010582010582)
    
    value = xai_ease_score(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.85)

    value = alpha_importance_score(xai_features.feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.38461538461538464)