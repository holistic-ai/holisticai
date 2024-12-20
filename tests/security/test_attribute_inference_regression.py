import numpy as np
from holisticai.datasets import load_dataset
from holisticai.security.attackers.attribute_inference.baseline import AttributeInferenceBaseline
from holisticai.security.attackers.attribute_inference.wrappers.regression.scikitlearn import ScikitlearnRegressor
from holisticai.security.attackers.attribute_inference.black_box import AttributeInferenceBlackBox
from holisticai.security.attackers.attribute_inference.true_label_baseline import AttributeInferenceBaselineTrueLabel
from holisticai.security.attackers.attribute_inference.wrappers.regression.scikitlearn import ScikitlearnDecisionTreeRegressor
from holisticai.security.attackers.attribute_inference.mitigation.attacks.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree
)
from holisticai.efficacy.metrics import classification_efficacy_metrics
from holisticai.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import pytest

np.random.seed(100)

SHARD_SIZE = 60

@pytest.fixture
def regression_dataset():
    dataset = load_dataset('us_crime', preprocessed=True, protected_attribute='race')
    dataset = dataset.sample(SHARD_SIZE, random_state=42)
    train_test = dataset.train_test_split(test_size=0.5, random_state=42)
    train = train_test['train']
    test = train_test['test']
    return train, test

def test_att_inf_baseline_regression(regression_dataset):
    train, test = regression_dataset
    train_data = train['X'].copy()
    train_data['group_a'] = train['group_a'].astype(int)

    test_data = test['X'].copy()
    test_data['group_a'] = test['group_a'].astype(int)

    x_train = train_data.values
    x_test = test_data.values

    y_train = train['y'].values
    y_test = test['y'].values

    attack_feature = 101 # last column represents the sensitive attribute

    attack = AttributeInferenceBaseline(attack_feature=attack_feature)
    attack.fit(x_train)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    feat_pred = attack.infer(attack_x_test, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5

def test_att_inf_blackbox_regression(regression_dataset):
    train, test = regression_dataset
    train_data = train['X'].copy()
    train_data['group_a'] = train['group_a'].astype(int)

    test_data = test['X'].copy()
    test_data['group_a'] = test['group_a'].astype(int)

    x_train = train_data.values
    x_test = test_data.values

    y_train = train['y'].values
    y_test = test['y'].values

    attack_feature = 101 # last column represents the sensitive attribute

    regressor = Pipeline(steps=[
        ('model', DecisionTreeRegressor())
    ])

    regressor.fit(x_train, y_train)

    # regressor = train_holisticai_regressor(x_train, y_train)
    regressor = ScikitlearnRegressor(regressor)
    pred = regressor.predict(x_train)

    attack = AttributeInferenceBlackBox(estimator=regressor, attack_feature=attack_feature, scale_range=(0,1))
    attack.fit(x_train, y_train, pred)

    pred = regressor.predict(x_test)
    attack_x_test = np.delete(x_test, attack_feature, axis=1)

    feat_true = x_test[:, attack_feature]

    values = [False, True]
    feat_pred = attack.infer(attack_x_test, y_test, pred, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_baseline_true_label_regression(regression_dataset):
    train, test = regression_dataset
    train_data = train['X'].copy()
    train_data['group_a'] = train['group_a'].astype(int)

    test_data = test['X'].copy()
    test_data['group_a'] = test['group_a'].astype(int)

    x_train = train_data.values
    x_test = test_data.values

    y_train = train['y'].values
    y_test = test['y'].values

    attack_feature = 101 # last column represents the sensitive attribute

    attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, is_regression=True)
    attack.fit(x_train, y_train)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    feat_pred = attack.infer(attack_x_test, y_test, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_white_box_lifestyle_decision_tree_regression(regression_dataset):
    train, test = regression_dataset
    train_data = train['X'].copy()
    train_data['group_a'] = train['group_a'].astype(int)

    test_data = test['X'].copy()
    test_data['group_a'] = test['group_a'].astype(int)

    x_train = train_data.values
    x_test = test_data.values

    y_train = train['y'].values
    y_test = test['y'].values

    attack_feature = 101 # last column represents the sensitive attribute

    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)

    regressor = ScikitlearnDecisionTreeRegressor(regressor)
    attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(estimator=regressor, attack_feature=attack_feature)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)

    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    priors = [2 / 30, 28 / 30]

    feat_pred = attack.infer(attack_x_test, values=values, priors=priors)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5