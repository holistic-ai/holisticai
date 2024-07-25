from sklearn.linear_model import LogisticRegression
from holisticai.utils.feature_importances import SurrogateFeatureImportanceCalculator, PermutationFeatureImportanceCalculator
from holisticai.utils import Importances
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
    from holisticai.utils import BinaryClassificationProxy
    proxy = BinaryClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=[0,1])
    fi_strategy = SurrogateFeatureImportanceCalculator(random_state=RandomState(42))
    importance = fi_strategy.compute_importances(test['X'], proxy=proxy)
    assert isinstance(importance, Importances)
    assert np.isclose(importance['capital-loss'], 0.37037037037037046, atol=5e-2)
    
def test_permutation_feature_importance_call(input_data):
    model, test = input_data
    from holisticai.utils import BinaryClassificationProxy
    proxy = BinaryClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=[0,1])
    fi_strategy = PermutationFeatureImportanceCalculator(random_state=RandomState(42))
    importance = fi_strategy.compute_importances(test, proxy=proxy)
    assert isinstance(importance, Importances)
    assert np.isclose(importance['capital-gain'], 0.38461538461538486, atol=5e-2)
    