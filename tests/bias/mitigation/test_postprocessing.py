import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_adult
from tests.testing_utils._tests_utils import data_info, evaluate_pipeline, fit


def check_postprocessing_prediction(model, data_info):
    train_data, test_data = data_info
    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    model.predict(X, **fit_params)
    model.predict_score(X, **fit_params)
    model.predictions(X, **fit_params)


def build_ceop_pipeline(cost_constraint):
    from holisticai.bias.mitigation import CalibratedEqualizedOdds

    np.random.seed(100)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
            (
                "bm_postprocessing",
                CalibratedEqualizedOdds(cost_constraint=cost_constraint),
            ),
        ]
    )
    return pipeline


def build_eop_pipeline():
    from holisticai.bias.mitigation import EqualizedOdds

    np.random.seed(100)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
            ("bm_postprocessing", EqualizedOdds()),
        ]
    )
    return pipeline


def build_roc_pipeline(metric_name):
    from holisticai.bias.mitigation import RejectOptionClassification

    np.random.seed(100)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
            ("bm_postprocessing", RejectOptionClassification(metric_name=metric_name)),
        ]
    )
    return pipeline


@pytest.mark.parametrize(
    "cost_constraint, metric_evalution",
    [
        ("fnr", (["False Negative Rate difference"], [0.05])),
        ("fpr", (["False Positive Rate difference"], [0.05])),
        ("weighted", (["False Positive Rate difference"], [0.05])),
    ],
)
def test_calibrated_equal_odds(data_info, cost_constraint, metric_evalution):
    pipeline = build_ceop_pipeline(cost_constraint)
    pipeline = fit(pipeline, data_info)
    evaluate_pipeline(pipeline, data_info, *metric_evalution)
    check_postprocessing_prediction(pipeline, data_info)


def test_equal_odds(data_info):
    pipeline = build_eop_pipeline()
    pipeline = fit(pipeline, data_info)
    evaluate_pipeline(pipeline, data_info, ["False Negative Rate difference"], [0.05])
    check_postprocessing_prediction(pipeline, data_info)


@pytest.mark.parametrize(
    "metric_name, metric_evalution",
    [
        ("Statistical parity difference", (["Statistical parity difference"], [0.05])),
        ("Average odds difference", (["Average odds difference"], [0.05])),
        ("Equal opportunity difference", (["Equal opportunity difference"], [0.05])),
    ],
)
def test_reject_option_classification(data_info, metric_name, metric_evalution):
    pipeline = build_roc_pipeline(metric_name=metric_name)
    pipeline = fit(pipeline, data_info)
    evaluate_pipeline(pipeline, data_info, *metric_evalution)
