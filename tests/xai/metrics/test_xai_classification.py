from holisticai.datasets import load_dataset
from sklearn.naive_bayes import GaussianNB
from holisticai.xai.metrics import classification_xai_metrics, classification_xai_features
from holisticai.xai.metrics.global_importance import alpha_importance_score, rank_alignment, position_parity, xai_ease_score
import numpy as np
import pytest

@pytest.fixture
def input_data():
    dataset = load_dataset('adult').select(list(range(1000)))
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']
    model = GaussianNB()
    model.fit(train['X'], train['y'])
    return model, test


@pytest.mark.parametrize("strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score", [
    ("permutation", 0.5, 0.5, 1.0, 0.012195121951219513),
    ("surrogate",  1.0, 1.0, 1.0, 0.012195121951219513)
])
def test_xai_classification_metrics(strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score, input_data):
    model, test = input_data
    metrics = classification_xai_metrics(test['X'], test['y'], model.predict, model.predict_proba, classes=[0,1], strategy=strategy)
    assert np.isclose(metrics.loc['Rank Alignment'].value, rank_alignment)
    assert np.isclose(metrics.loc['Position Parity'].value, position_parity)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, xai_ease_score)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, alpha_imp_score)


def test_xai_classification_metrics_separated(input_data):
    model, test = input_data

    xai_features = classification_xai_features(test["X"], test["y"], model.predict, model.predict_proba, classes=[0,1])
        
    value = rank_alignment(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.5)

    value = position_parity(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.5)
    
    value = xai_ease_score(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    assert np.isclose(value, 1.0)

    value = alpha_importance_score(xai_features.feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 0.012195121951219513)