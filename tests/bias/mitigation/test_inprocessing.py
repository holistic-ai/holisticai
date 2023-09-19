import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import evaluate_pipeline, fit, small_categorical_dataset

from holisticai.bias.mitigation import GridSearchReduction
from holisticai.pipeline import Pipeline


def build_gsr_pipeline():
    model = LogisticRegression()
    inprocessing_model = GridSearchReduction(
        constraints="EqualizedOdds", grid_size=5
    ).transform_estimator(model)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    return pipeline


def test_GridSearchReduction(small_categorical_dataset):
    np.random.seed(100)
    pipeline = build_gsr_pipeline()
    pipeline = fit(pipeline, small_categorical_dataset)
    evaluate_pipeline(
        pipeline, small_categorical_dataset, ["Statistical parity difference"], [0.05]
    )
