from sklearn.naive_bayes import GaussianNB
from holisticai.datasets import load_dataset   
from holisticai.explainability.metrics import multiclass_explainability_metrics
from holisticai.explainability.metrics.global_importance import alpha_score, rank_alignment, position_parity, xai_ease_score
import numpy as np
import pytest

@pytest.fixture
def input_data():
    dataset = load_dataset('student_multiclass').select(list(range(1000)))
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']

    model = GaussianNB()
    model.fit(train['X'], train['y'])
    return model, test


def get_multiclass_features(model, test):
    from holisticai.utils import MultiClassificationProxy
    from holisticai.utils.feature_importances import compute_permutation_feature_importance
    from holisticai.utils.inspection import compute_partial_dependence
    
    proxy = MultiClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=model.classes_)
    importances  = compute_permutation_feature_importance(proxy=proxy, X=test['X'], y=test['y'])
    ranked_importances = importances.top_alpha(0.8)
    partial_dependencies = compute_partial_dependence(test['X'], features=ranked_importances.feature_names, proxy=proxy)
    conditional_importances  = compute_permutation_feature_importance(proxy=proxy, X=test['X'], y=test['y'], conditional=True)
    return importances, ranked_importances, conditional_importances, partial_dependencies



@pytest.mark.skip(reason="Sometimes fails due to randomization of the dataset.")
@pytest.mark.parametrize("strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score", [
    ("permutation", 0.4743650793650794, 0.129973544973545, 0.8833333333333333, 0.38461538461538464),
    ("surrogate",  0.2962962962962963, 0.2962962962962963, 0.7222222222222222, 0.11538461538461539)
])
def test_xai_multiclassification_metrics(strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score, input_data):
    model, test = input_data
    metrics = multiclass_explainability_metrics(test['X'], test['y'], model.predict, model.predict_proba, classes=[0,1,2], strategy=strategy)
    assert np.isclose(metrics.loc['Rank Alignment'].value, rank_alignment)
    assert np.isclose(metrics.loc['Position Parity'].value, position_parity)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, xai_ease_score)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, alpha_imp_score)

@pytest.mark.skip(reason="Sometimes fails due to randomization of the dataset.")
def test_xai_classification_metrics_separated(input_data):
    model, test = input_data
    
    importances, ranked_importances, conditional_importances, partial_dependencies = get_multiclass_features(model, test)
        
    value = rank_alignment(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.4743650793650794)

    value = position_parity(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.129973544973545)
    
    value = xai_ease_score(partial_dependencies, ranked_importances)
    assert np.isclose(value, 0.8833333333333333)

    value = alpha_score(importances)
    assert np.isclose(value, 0.38461538461538464)