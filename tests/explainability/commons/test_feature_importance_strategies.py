from sklearn.linear_model import LogisticRegression
from holisticai.explainability.commons import SurrogateFeatureImportanceCalculator, PermutationFeatureImportanceCalculator, Importances
from holisticai.datasets import load_dataset
import pytest
import numpy as np
from numpy.random import RandomState

@pytest.fixture
def input_data():
    dataset = load_dataset('adult').sample(n=100, random_state=42)
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    train = dataset['test']
    test = dataset['test']
    
    model = LogisticRegression(random_state=42)
    model.fit(train['X'], train['y'])
    return model, test

def test_surrogate_feature_importance_call(input_data):
    model, test = input_data
    from holisticai.explainability.commons._definitions import BinaryClassificationXAISettings
    from holisticai.datasets import Dataset
    learning_task_settings = BinaryClassificationXAISettings(predict_fn=model.predict, predict_proba_fn=model.predict_proba, classes=[0,1])
    fi_strategy = SurrogateFeatureImportanceCalculator(learning_task_settings=learning_task_settings, random_state=RandomState(42))
    importance = fi_strategy(test)
    assert isinstance(importance, Importances)
    assert np.isclose(importance['education-num'], 0.5263157894736843)
    
def test_permutation_feature_importance_call(input_data):
    model, test = input_data
    from holisticai.explainability.commons._definitions import BinaryClassificationXAISettings
    learning_task_settings = BinaryClassificationXAISettings(predict_fn=model.predict, predict_proba_fn=model.predict_proba, classes=[0,1])
    fi_strategy = PermutationFeatureImportanceCalculator(learning_task_settings=learning_task_settings, random_state=RandomState(42))
    importance = fi_strategy(test)
    assert isinstance(importance, Importances)
    assert np.isclose(importance['education-num'], 0.10526315789473678)
    