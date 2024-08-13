
import warnings
warnings.filterwarnings("ignore")

from holisticai.datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from holisticai.robustness.attackers import HopSkipJump, ZooAttack
from holisticai.utils import BinaryClassificationProxy
from holisticai.robustness.attackers.classification.commons import format_function_predict_proba
from holisticai.robustness.metrics import adversarial_accuracy, empirical_robustness
from sklearn.metrics import accuracy_score
import pytest

SHARD_SIZE=60

@pytest.fixture
def categorical_dataset():
    dataset = load_dataset("adult", protected_attribute='race') # x,y,group_a,group_b
    dataset = dataset.groupby(['y','group_a']).sample(SHARD_SIZE, random_state=42) # 0-ga | 0-gb  | 1-ga | 1-gb
    train_test = dataset.train_test_split(test_size=0.02, stratify=dataset['y'], random_state=0)
    train = train_test['train']
    test = train_test['test']
    correlations = train['X'].corrwith(train['y']).sort_values(ascending=False)
    top_10_features = correlations.head(10).index.tolist()
    train['X'] = train['X'][top_10_features]
    test['X'] = test['X'][top_10_features]
    return train, test


def test_hsj(categorical_dataset):
    train, test = categorical_dataset

    # Standardize data and fit model
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    pipe.fit(train['X'], train['y'])

    y_pred = pipe.predict(test['X'])
    baseline_accuracy = accuracy_score(test['y'], y_pred)

    proxy = BinaryClassificationProxy(predict=pipe.predict, predict_proba=pipe.predict_proba, classes=[0, 1])

    hsj_attacker = HopSkipJump(name="HSJ", predictor=proxy.predict)

    hsj_adv_x = hsj_attacker.generate(test['X'])
    y_adv_pred = proxy.predict(hsj_adv_x)

    hsj_accuracy = adversarial_accuracy(test['y'], y_pred, y_adv_pred)
    hsj_robustness = empirical_robustness(test['X'], hsj_adv_x, y_pred, y_adv_pred, norm=2)


    # This test should pass when both dataframes are different
    assert not hsj_adv_x.equals(test['X'])
    # This test should pass when both accuracies are different
    assert hsj_accuracy != baseline_accuracy
    # This test should pass when the robustness is greater than 0
    assert hsj_robustness > 0

def test_zoo(categorical_dataset):
    train, test = categorical_dataset

    # Standardize data and fit model
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    pipe.fit(train['X'], train['y'])

    y_pred = pipe.predict(test['X'])
    baseline_accuracy = accuracy_score(test['y'], y_pred)

    proxy = BinaryClassificationProxy(predict=pipe.predict, predict_proba=pipe.predict_proba, classes=[0, 1])

    zoo_attacker = ZooAttack(name="Zoo", proxy=proxy)

    zoo_adv_x = zoo_attacker.generate(test['X'])

    y_adv_pred = proxy.predict(zoo_adv_x)
    zoo_accuracy = adversarial_accuracy(test['y'], y_pred, y_adv_pred)
    zoo_robustness = empirical_robustness(test['X'], zoo_adv_x, y_pred, y_adv_pred, norm=2)

    # This test should pass when both dataframes are different
    assert not zoo_adv_x.equals(test['X'])
    # This test should pass when both accuracies are different
    assert zoo_accuracy != baseline_accuracy
    # This test should pass when the robustness is greater than 0
    assert zoo_robustness > 0
