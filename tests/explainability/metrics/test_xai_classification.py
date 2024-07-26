from holisticai.datasets import load_dataset
from sklearn.naive_bayes import GaussianNB
from holisticai.explainability.metrics import classification_explainability_metrics
from holisticai.explainability.metrics.global_importance import alpha_score, rank_alignment, position_parity, xai_ease_score
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

def get_classification_features(model, test, strategy):
    from holisticai.utils import BinaryClassificationProxy
    from holisticai.utils.inspection import compute_partial_dependence

    if strategy=='permutation':
        from holisticai.utils.feature_importances import compute_permutation_feature_importance as compute_feature_importance
    elif strategy=='surrogate':
        from holisticai.utils.feature_importances import compute_surrogate_feature_importance as compute_feature_importance
    else:
        raise ValueError("Invalid strategy")
    
    
    proxy = BinaryClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=model.classes_)
    importances  = compute_feature_importance(proxy=proxy, X=test['X'], y=test['y'])
    ranked_importances = importances.top_alpha(0.8)
    partial_dependencies = compute_partial_dependence(test['X'], features=ranked_importances.feature_names, proxy=proxy)
    conditional_importances  = compute_feature_importance(proxy=proxy, X=test['X'], y=test['y'], conditional=True)
    return proxy, importances, ranked_importances, conditional_importances, partial_dependencies


@pytest.mark.parametrize("strategy, alpha_imp_score, xai_ease_score, position_parity, rank_alignment", [
    ("permutation", 0.010309278350515464, 1.0, 0.0, 0.0),
    ("surrogate",  0.010309278350515464, 1.0, 1.0, 1.0)
])
def test_xai_classification_metrics(strategy, alpha_imp_score, xai_ease_score, position_parity, rank_alignment, input_data):
    model, test = input_data
    proxy, importances, ranked_importances, conditional_importances, partial_dependencies = get_classification_features(model, test, strategy)

    metrics = classification_explainability_metrics(importances, partial_dependencies, conditional_importances, X=test['X'], y_pred=proxy.predict(test['X']))
    assert np.isclose(metrics.loc['Rank Alignment'].value, rank_alignment)
    assert np.isclose(metrics.loc['Position Parity'].value, position_parity)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, xai_ease_score)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, alpha_imp_score)


def test_xai_classification_metrics_separated(input_data):
    model, test = input_data

    proxy, importances, ranked_importances, conditional_importances, partial_dependencies = get_classification_features(model, test, 'permutation')
        
    value = rank_alignment(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.0)

    value = position_parity(conditional_importances, ranked_importances)
    assert np.isclose(value, 0.0)
    
    value = xai_ease_score(partial_dependencies, ranked_importances)
    assert np.isclose(value, 1.0)

    value = alpha_score(importances)
    assert np.isclose(value, 0.010309278350515464)


def test_rank_alignment():
    from holisticai.explainability.metrics import rank_alignment
    from holisticai.utils import ConditionalImportances, Importances

    values = {
        '0': Importances(values=[0.1, 0.2, 0.3, 0.4], 
                         feature_names=['feature_2', 'feature_3', 'feature_4']),
        '1': Importances(values=[0.4, 0.3, 0.2, 0.1], 
                         feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    }
    conditional_feature_importance = ConditionalImportances(values=values)

    ranked_feature_importance = Importances(values=[0.5, 0.3, 0.2], feature_names=['feature_1', 'feature_2', 'feature_3'])
    score = rank_alignment(conditional_feature_importance, ranked_feature_importance)
    assert np.isclose(score,0.6944444444444444)

def test_position_parity():
    from holisticai.utils import ConditionalImportances, Importances
    from holisticai.explainability.metrics import position_parity
    import numpy as np

    values = np.array([0.50, 0.40, 0.10])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)

    values = {
    "group1": Importances(values=np.array([0.40, 0.35, 0.25]),
                           feature_names=["feature_1", "feature_2", "feature_3"]),
    "group2": Importances(values=np.array([0.50, 0.30, 0.20]),
                           feature_names=["feature_3", "feature_2", "feature_1"]),
    }
    conditional_feature_importance = ConditionalImportances(values=values)
    score = position_parity(conditional_feature_importance, feature_importance)
    assert np.isclose(score,0.6388888888888888)

def test_alpha_score():
    from holisticai.utils import Importances
    from holisticai.explainability.metrics import alpha_score

    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = alpha_score(feature_importance)
    assert np.isclose(score, 0.6666666666666666)

def test_xai_ease_score():
    from holisticai.utils import PartialDependence, Importances
    from holisticai.explainability.metrics.global_importance import xai_ease_score
    
    partial_dependence = [[
        {
            'average': [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]],
            'grid_values': [[1,2,3,4,5,6,7,8,9]]
        },
        {
            'average': [[0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6]], 
            'grid_values': [[1,2,3,4,5,6,7,8,9]]
        },
    ]]
    partial_dependence = PartialDependence(values=partial_dependence)
    feature_importance = Importances(values=np.array([0.5, 0.5]),
    feature_names=['feature1', 'feature2'])
    score = xai_ease_score(partial_dependence, feature_importance)
    assert np.isclose(score, 0.5)

def test_spread_ratio():
    from holisticai.utils import Importances
    from holisticai.explainability.metrics.global_importance import spread_ratio
    
    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = spread_ratio(feature_importance)
    assert np.isclose(score, 0.9206198357143052)

def test_spread_divergence():
    from holisticai.utils import Importances
    from holisticai.explainability.metrics.global_importance import spread_divergence
    
    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = spread_divergence(feature_importance)
    assert np.isclose(score, 0.8196393599933761)