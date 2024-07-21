from holisticai.datasets import load_dataset
from sklearn.naive_bayes import GaussianNB
from holisticai.explainability.metrics import classification_explainability_metrics, classification_explainability_features
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


@pytest.mark.parametrize("strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score", [
    ("permutation", 1.0, 1.0, 1.0, 0.012195121951219513),
    ("surrogate",  1.0, 1.0, 1.0, 0.012195121951219513)
])
def test_xai_classification_metrics(strategy, rank_alignment, position_parity, xai_ease_score, alpha_imp_score, input_data):
    model, test = input_data
    metrics = classification_explainability_metrics(test['X'], test['y'], model.predict, model.predict_proba, classes=[0,1], strategy=strategy)
    assert np.isclose(metrics.loc['Rank Alignment'].value, rank_alignment)
    assert np.isclose(metrics.loc['Position Parity'].value, position_parity)
    assert np.isclose(metrics.loc['XAI Ease Score'].value, xai_ease_score)
    assert np.isclose(metrics.loc['Alpha Importance Score'].value, alpha_imp_score)


def test_xai_classification_metrics_separated(input_data):
    model, test = input_data

    xai_features = classification_explainability_features(test["X"], test["y"], model.predict, model.predict_proba, classes=[0,1])
        
    value = rank_alignment(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 1.0)

    value = position_parity(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    assert np.isclose(value, 1.0)
    
    value = xai_ease_score(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    assert np.isclose(value, 1.0)

    value = alpha_score(xai_features.feature_importance)
    assert np.isclose(value, 0.012195121951219513)


def test_rank_alignment():
    from holisticai.explainability.metrics import rank_alignment
    from holisticai.explainability.commons import ConditionalFeatureImportance, Importances

    values = {
        '0': Importances(values=[0.1, 0.2, 0.3, 0.4], 
                         feature_names=['feature_2', 'feature_3', 'feature_4']),
        '1': Importances(values=[0.4, 0.3, 0.2, 0.1], 
                         feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    }
    conditional_feature_importance = ConditionalFeatureImportance(values=values)

    ranked_feature_importance = Importances(values=[0.5, 0.3, 0.2], feature_names=['feature_1', 'feature_2', 'feature_3'])
    score = rank_alignment(conditional_feature_importance, ranked_feature_importance)
    assert np.isclose(score,0.6944444444444444)

def test_position_parity():
    from holisticai.explainability.commons import ConditionalFeatureImportance, Importances
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
    conditional_feature_importance = ConditionalFeatureImportance(values=values)
    score = position_parity(conditional_feature_importance, feature_importance)
    assert np.isclose(score,0.6388888888888888)

def test_alpha_score():
    from holisticai.explainability.commons import Importances
    from holisticai.explainability.metrics import alpha_score

    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = alpha_score(feature_importance)
    assert np.isclose(score, 0.6666666666666666)

def test_xai_ease_score():
    from holisticai.explainability.commons import PartialDependence, Importances
    from holisticai.explainability.metrics.global_importance import xai_ease_score
    
    partial_dependence = [
        {
            'average': [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]],
            'grid_values': [[1,2,3,4,5,6,7,8,9]]
        },
        {
            'average': [[0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6]], 
            'grid_values': [[1,2,3,4,5,6,7,8,9]]
        },
    ]
    partial_dependence = PartialDependence(values=partial_dependence)
    feature_importance = Importances(values=np.array([0.5, 0.5]),
    feature_names=['feature1', 'feature2'])
    score = xai_ease_score(partial_dependence, feature_importance)
    assert np.isclose(score, 0.5)

def test_spread_ratio():
    from holisticai.explainability.commons import Importances
    from holisticai.explainability.metrics.global_importance import spread_ratio
    
    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = spread_ratio(feature_importance)
    assert np.isclose(score, 0.9206198357143052)

def test_spread_divergence():
    from holisticai.explainability.commons import Importances
    from holisticai.explainability.metrics.global_importance import spread_divergence
    
    values = np.array([0.10, 0.20, 0.30])
    feature_names = ["feature_1", "feature_2", "feature_3"]
    feature_importance = Importances(values=values, feature_names=feature_names)
    score = spread_divergence(feature_importance)
    assert np.isclose(score, 0.8196393599933761)