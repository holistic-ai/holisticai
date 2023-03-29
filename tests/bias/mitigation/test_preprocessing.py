import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import evaluate_pipeline, fit, small_categorical_dataset

from holisticai.pipeline import Pipeline


def build_rew_pipeline():
    from holisticai.bias.mitigation import Reweighing

    np.random.seed(100)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", Reweighing()),
            ("classifier", LogisticRegression()),
        ]
    )
    return pipeline


def test_reweighing(small_categorical_dataset):
    pipeline = build_rew_pipeline()
    pipeline = fit(pipeline, small_categorical_dataset)
    evaluate_pipeline(
        pipeline, small_categorical_dataset, ["Statistical parity difference"], [0.05]
    )
