
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from holisticai.datasets import load_dataset
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
    return train['X'], test['X'], train['y'], test['y']


def test_hsj(categorical_dataset):
    train_X, test_X, train_y, test_y = categorical_dataset

    # Standardize data and fit model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)

    X_test = scaler.transform(test_X)
    feature_names = list(test_X.columns)

    y_train = train_y
    y_test = test_y

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    proxy = BinaryClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=[0, 1])

    kargs = {
        "predictor": proxy.predict,
        "input_shape": tuple(X_test.shape[1:]),
    }
    hsj_attacker = HopSkipJump(name="HSJ", **kargs)

    hsj_adv_x = hsj_attacker.generate(pd.DataFrame(X_test, columns=feature_names))
    y_adv_pred = proxy.predict(hsj_adv_x)

    hsj_accuracy = adversarial_accuracy(y_test, y_pred, y_adv_pred)
    hsj_robustness = empirical_robustness(X_test, hsj_adv_x, y_pred, y_adv_pred, norm=2)

    # This test should pass when both dataframes are different
    assert not hsj_adv_x.equals(pd.DataFrame(X_test, columns=feature_names))
    # This test should pass when both accuracies are different
    assert hsj_accuracy != baseline_accuracy
    # This test should pass when the robustness is greater than 0
    assert hsj_robustness > 0

def test_zoo(categorical_dataset):
    train_X, test_X, train_y, test_y = categorical_dataset

    # Standardize data and fit model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_X)

    X_test = scaler.transform(test_X)
    feature_names = list(test_X.columns)

    y_train = train_y
    y_test = test_y

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    proxy = BinaryClassificationProxy(predict=model.predict, predict_proba=model.predict_proba, classes=[0, 1])

    predict_proba_fn = format_function_predict_proba(proxy.learning_task, proxy.predict_proba)  # type: ignore

    kargs = {
        "predict_proba_fn": predict_proba_fn,
        "input_shape": tuple(X_test.shape[1:]),
    }
    zoo_attacker = ZooAttack(name="Zoo", **kargs)

    zoo_adv_x = zoo_attacker.generate(pd.DataFrame(X_test, columns=feature_names))

    y_adv_pred = proxy.predict(zoo_adv_x)
    zoo_accuracy = adversarial_accuracy(y_test, y_pred, y_adv_pred)
    zoo_robustness = empirical_robustness(X_test, zoo_adv_x, y_pred, y_adv_pred, norm=2)

    # This test should pass when both dataframes are different
    assert not zoo_adv_x.equals(pd.DataFrame(X_test, columns=feature_names))
    # This test should pass when both accuracies are different
    assert zoo_accuracy != baseline_accuracy
    # This test should pass when the robustness is greater than 0
    assert zoo_robustness > 0
