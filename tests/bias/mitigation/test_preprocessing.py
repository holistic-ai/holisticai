import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_adult
from tests.testing_utils._tests_utils import data_info, evaluate_pipeline, fit


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


def test_reweighing(data_info):
    pipeline = build_rew_pipeline()
    pipeline = fit(pipeline, data_info)
    evaluate_pipeline(pipeline, data_info, ["Statistical parity difference"], [0.05])
