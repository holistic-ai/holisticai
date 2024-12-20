import numpy as np
from holisticai.datasets import load_dataset
from holisticai.security.attackers.attribute_inference.baseline import AttributeInferenceBaseline

from holisticai.security.attackers.attribute_inference.wrappers.classification.scikitlearn import SklearnClassifier
from holisticai.security.attackers.attribute_inference.black_box import AttributeInferenceBlackBox
from holisticai.security.attackers.attribute_inference.true_label_baseline import AttributeInferenceBaselineTrueLabel
from holisticai.security.attackers.attribute_inference.wrappers.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from holisticai.security.attackers.attribute_inference.mitigation.attacks.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree
)
from holisticai.security.attackers.attribute_inference.mitigation.scikitlearn import (
    AttributeInferenceWhiteBoxDecisionTree,
)
from sklearn.tree import DecisionTreeClassifier
from holisticai.efficacy.metrics import classification_efficacy_metrics
import pytest

np.random.seed(100)

SHARD_SIZE=60

@pytest.fixture
def categorical_dataset():
    dataset = load_dataset("adult", protected_attribute='race') # x,y,group_a,group_b
    dataset = dataset.groupby(['y','group_a']).sample(SHARD_SIZE, random_state=42) # 0-ga | 0-gb  | 1-ga | 1-gb
    train_test = dataset.train_test_split(test_size=0.5, stratify=dataset['y'], random_state=0)
    train = train_test['train']
    test = train_test['test']
    correlations = train['X'].corrwith(train['y']).sort_values(ascending=False)
    top_10_features = correlations.head(10).index.tolist()
    train['X'] = train['X'][top_10_features]
    test['X'] = test['X'][top_10_features]
    return train, test


def test_att_inf_baseline(categorical_dataset):

    train, test = categorical_dataset

    attack_feature = 0 # marital status

    x_train = train['X'].copy().values
    y_train = train['y'].copy().values
    x_test = test['X'].copy().values
    y_test = test['y'].copy().values

    attack = AttributeInferenceBaseline(attack_feature=attack_feature)
    attack.fit(x_train)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    feat_pred = attack.infer(attack_x_test, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_black_box(categorical_dataset):
    train, test = categorical_dataset

    attack_feature = 0 # marital status

    x_train = train['X'].copy().values
    y_train = train['y'].copy().values
    x_test = test['X'].copy().values
    y_test = test['y'].copy().values

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    classifier = SklearnClassifier(classifier)

    attack = AttributeInferenceBlackBox(estimator=classifier, attack_feature=attack_feature)

    pred = classifier.predict_proba(x_train)
    attack.fit(x_train, y_train, pred)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    pred = classifier.predict_proba(x_test)

    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    feat_pred = attack.infer(attack_x_test, y_test, pred, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_truelabel(categorical_dataset):
    train, test = categorical_dataset

    attack_feature = 0 # marital status

    x_train = train['X'].copy().values
    y_train = train['y'].copy().values
    x_test = test['X'].copy().values
    y_test = test['y'].copy().values

    attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature)
    attack.fit(x_train, y_train)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    feat_pred = attack.infer(attack_x_test, y_test, values=values)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_white_box_lifestyle(categorical_dataset):
    train, test = categorical_dataset

    attack_feature = 0 # marital status

    x_train = train['X'].copy().values
    y_train = train['y'].copy().values
    x_test = test['X'].copy().values
    y_test = test['y'].copy().values

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    classifier = ScikitlearnDecisionTreeClassifier(classifier)

    attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(attack_feature=attack_feature, estimator=classifier)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    priors = [53/120, 67/120]

    feat_pred = attack.infer(attack_x_test, values=values, priors=priors)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5


def test_att_inf_white_box(categorical_dataset):
    train, test = categorical_dataset

    attack_feature = 0 # marital status

    x_train = train['X'].copy().values
    y_train = train['y'].copy().values
    x_test = test['X'].copy().values
    y_test = test['y'].copy().values

    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    classifier = ScikitlearnDecisionTreeClassifier(classifier)

    attack = AttributeInferenceWhiteBoxDecisionTree(attack_feature=attack_feature, classifier=classifier)

    attack_x_test = np.delete(x_test, attack_feature, axis=1)
    feat_true = x_test[:, attack_feature]

    values = [0, 1]
    priors = [53/120, 67/120]

    feat_pred = attack.infer(attack_x_test, y_test, values=values, priors=priors)

    assert len(feat_true) == len(feat_pred)

    df = classification_efficacy_metrics(feat_true, feat_pred)

    assert df.loc['Accuracy']['Value'] > 0.5