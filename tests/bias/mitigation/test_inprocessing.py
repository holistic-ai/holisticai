import os

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.mitigation import GridSearchReduction
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    evaluate_pipeline,
    fit,
    small_categorical_dataset,
)


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
        pipeline, small_categorical_dataset, ["Statistical parity difference"], [0.1]
    )
