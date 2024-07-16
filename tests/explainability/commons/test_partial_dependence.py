import pytest
from holisticai.explainability.commons._feature_importance import compute_ranked_feature_importance
from holisticai.explainability.commons._partial_dependence import compute_partial_dependence
from holisticai.explainability.commons import BinaryClassificationXAISettings
from holisticai.explainability.commons import PermutationFeatureImportanceCalculator
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

    learning_task_settings = BinaryClassificationXAISettings(predict_fn=model.predict, predict_proba_fn=model.predict_proba, classes=[0,1])
    feature_importance_fn = PermutationFeatureImportanceCalculator(learning_task_settings=learning_task_settings, random_sate=RandomState(42))
    feature_importance = feature_importance_fn(test)
    filtered_feature_importance = compute_ranked_feature_importance(feature_importance)
    return train['X'], learning_task_settings, filtered_feature_importance.feature_names

def test_get_partial_dependence_results_binary_classification(input_data):
    x, learning_task_settings, important_features = input_data
    results = compute_partial_dependence(x, important_features, learning_task_settings)
    np.isclose(results[0].values[0]['average'][0][0], 0.7463041031804294)