import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.bias.mitigation import FairScoreClassifier
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime


def test_fair_scoring_classifier():
    train_data, test_data = load_preprocessed_us_crime(nb_classes=5)
    X, y, group_a, group_b = train_data
    X_test, y_test, group_a_test, group_b_test = test_data
    y = pd.get_dummies(y).values

    objectives = "ba"
    constraints = {}

    numUsers = y_test.shape

    model = FairScoreClassifier(objectives, constraints)

    model.fit(X, y, group_a, group_b)

    assert model.predict(X_test, group_a_test, group_b_test).shape == (numUsers)
