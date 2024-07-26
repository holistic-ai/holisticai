import pytest
from holisticai.datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState

@pytest.fixture
def input_data():
    dataset = load_dataset("adult")
    dataset = dataset.sample(n=1000, random_state=42)
    dataset = dataset.train_test_split(test_size=0.3)
    train = dataset['train']
    test = dataset['test']

    model = LogisticRegression(random_state=42)
    model.fit(train['X'], train['y'])

    return test, model

def test_get_partial_dependence_results_binary_classification(input_data):
    test,model = input_data
    from holisticai.utils import RegressionProxy
    from holisticai.utils.feature_importances import compute_permutation_feature_importance
    from holisticai.utils.inspection import compute_partial_dependence
    
    proxy = RegressionProxy(predict=model.predict)
    importances  = compute_permutation_feature_importance(proxy=proxy, X=test['X'], y=test['y'])
    ranked_importances = importances.top_alpha(0.8)
    partial_dependencies = compute_partial_dependence(test['X'], features=ranked_importances.feature_names, proxy=proxy)
    np.isclose(partial_dependencies.values[0][0]['average'][0][0], 0.03)