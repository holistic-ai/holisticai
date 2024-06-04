import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from holisticai.mitigation.bias import FairScoreClassifier
from tests.bias.mitigation.testing_utils.utils import small_multiclass_dataset as ds


def test_fair_scoring_classifier(ds):
    train = ds['train']
    test = ds['test']
    objectives = "ba"

    constraints = {}

    numUsers = test['y'].shape

    model = FairScoreClassifier(objectives, constraints)
    model.fit(train['x'], train['y'], train['group_a'], train['group_b'])

    assert model.predict(test['x'], test['group_a'], test['group_b']).shape == (numUsers)
