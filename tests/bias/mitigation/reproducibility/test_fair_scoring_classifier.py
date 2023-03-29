import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from testing_utils.tests_utils import small_multiclass_dataset

from holisticai.bias.mitigation import FairScoreClassifier


def test_fair_scoring_classifier(small_multiclass_dataset):
    train_data, test_data = small_multiclass_dataset
    X, y, group_a, group_b = train_data
    X_test, y_test, group_a_test, group_b_test = test_data
    y = pd.get_dummies(y).values

    objectives = "ba"

    constraints = {}

    numUsers = y_test.shape

    model = FairScoreClassifier(objectives, constraints)

    model.fit(X, y, group_a, group_b)

    assert model.predict(X_test, group_a_test, group_b_test).shape == (numUsers)
