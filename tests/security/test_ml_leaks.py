import pytest
import numpy as np
from sklearn.base import BaseEstimator
from holisticai.security.attackers import MLleaks
from sklearn.ensemble import RandomForestClassifier

from holisticai.datasets import load_dataset


SHARD_SIZE=100

@pytest.fixture
def tgt_shw_dataset():
    dataset = load_dataset("adult", protected_attribute='sex') # x,y,group_a,group_b
    dataset = dataset.groupby(['y','group_a']).sample(SHARD_SIZE, random_state=42) # 0-ga | 0-gb  | 1-ga | 1-gb
    train_test = dataset.train_test_split(test_size=0.5, stratify=dataset['y'], random_state=0)
    target = train_test['train'].train_test_split(test_size=0.5, random_state=0)
    shadow = train_test['test'].train_test_split(test_size=0.5, random_state=0)
    X_target_train = target['train']['X']
    y_target_train = target['train']['y']
    X_target_test = target['test']['X']
    y_target_test = target['test']['y']

    X_shadow_train = shadow['train']['X']
    y_shadow_train = shadow['train']['y']
    X_shadow_test = shadow['test']['X']
    y_shadow_test = shadow['test']['y']

    tgt_dataset = ((X_target_train, y_target_train), (X_target_test, y_target_test))
    sdw_dataset = ((X_shadow_train, y_shadow_train), (X_shadow_test, y_shadow_test))
    return tgt_dataset, sdw_dataset 



class MockModel(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return np.random.rand(len(X), 2)


@pytest.fixture
def ml_leaks_instance(tgt_shw_dataset):
    target_dataset, shadow_dataset = tgt_shw_dataset
    X_target_train, y_target_train = target_dataset[0]
    target_model = RandomForestClassifier(random_state=42)
    target_model.fit(X_target_train, y_target_train)
    return MLleaks(target_model, target_dataset, shadow_dataset)


@pytest.fixture
def ml_leaks_no_clone_instance(tgt_shw_dataset):
    target_dataset, shadow_dataset = tgt_shw_dataset
    X_target_train, y_target_train = target_dataset[0]
    target_model = RandomForestClassifier(random_state=42)
    target_model.fit(X_target_train, y_target_train)
    return MLleaks(target_model, target_dataset, shadow_dataset, clone_model=True)


def test_generate_attack_dataset(ml_leaks_instance):
    train_data, test_data = ml_leaks_instance.generate_attack_dataset()
    assert len(train_data) == 2
    assert len(test_data) == 2
    assert train_data[0].shape[0] == train_data[1].shape[0]
    assert test_data[0].shape[0] == test_data[1].shape[0]


def test_fit(ml_leaks_instance):
    ml_leaks_instance.generate_attack_dataset()
    attacker_model = ml_leaks_instance.fit()
    assert attacker_model is not None


def test_get_probs(ml_leaks_instance, tgt_shw_dataset):
    target_dataset, _ = tgt_shw_dataset
    model = ml_leaks_instance.target_model
    X_train, _ = target_dataset[0]
    X_test, _ = target_dataset[1]
    train_probs, test_probs = ml_leaks_instance._get_probs(model, X_train, X_test)
    assert train_probs.shape[0] == X_train.shape[0]
    assert test_probs.shape[0] == X_test.shape[0]


def test_create_attacker_dataset(ml_leaks_instance):
    train_data, test_data = ml_leaks_instance.generate_attack_dataset()
    X_mia_train, y_mia_train = train_data
    X_mia_test, y_mia_test = test_data
    assert X_mia_train.shape[0] == 200
    assert X_mia_test.shape[0] == 200
    assert np.sum(y_mia_train) == 100  # 100 training samples labeled as 1
    assert np.sum(y_mia_test) == 100  # 100 testing samples labeled as 1


def test_create_attacker_dataset_no_clone(ml_leaks_no_clone_instance):
    train_data, test_data = ml_leaks_no_clone_instance.generate_attack_dataset()
    X_mia_train, y_mia_train = train_data
    X_mia_test, y_mia_test = test_data
    assert X_mia_train.shape[0] == 600
    assert X_mia_test.shape[0] == 200
    assert np.sum(y_mia_train) == 300  # 100 training samples labeled as 1
    assert np.sum(y_mia_test) == 100  # 100 testing samples labeled as 1
