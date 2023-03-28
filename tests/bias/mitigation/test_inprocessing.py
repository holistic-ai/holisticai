import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_adult
from tests.testing_utils._tests_utils import data_info, evaluate_pipeline, fit


def build_gsr_pipeline():
    from holisticai.bias.mitigation import GridSearchReduction

    np.random.seed(100)

    model = LogisticRegression()
    inprocessing_model = GridSearchReduction(
        constraints="EqualizedOdds", grid_size=20
    ).transform_estimator(model)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    return pipeline


def test_reweighing(data_info):
    pipeline = build_gsr_pipeline()
    pipeline = fit(pipeline, data_info)
    evaluate_pipeline(pipeline, data_info, ["Statistical parity difference"], [0.05])
