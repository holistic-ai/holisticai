import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    evaluate_pipeline,
    fit,
    small_categorical_dataset,
)


def check_postprocessing_prediction(model, small_categorical_dataset):
    train_data, _ = small_categorical_dataset
    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    model.predict(X, **fit_params)
    model.predict_score(X, **fit_params)
    model.predictions(X, **fit_params)


def build_ceop_pipeline(cost_constraint):
    from holisticai.mitigation.bias import CalibratedEqualizedOdds

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
    from holisticai.mitigation.bias import EqualizedOdds

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
    from holisticai.mitigation.bias import RejectOptionClassification

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
        ("fnr", (["False Negative Rate difference"], [0.1])),
        ("fpr", (["False Positive Rate difference"], [0.5])),
        ("weighted", (["False Positive Rate difference"], [0.5])),
    ],
)
def test_calibrated_equal_odds(
    small_categorical_dataset, cost_constraint, metric_evalution
):
    pipeline = build_ceop_pipeline(cost_constraint)
    pipeline = fit(pipeline, small_categorical_dataset)
    evaluate_pipeline(pipeline, small_categorical_dataset, *metric_evalution)
    check_postprocessing_prediction(pipeline, small_categorical_dataset)


def test_equal_odds(small_categorical_dataset):
    pipeline = build_eop_pipeline()
    pipeline = fit(pipeline, small_categorical_dataset)
    evaluate_pipeline(
        pipeline, small_categorical_dataset, ["False Negative Rate difference"], [0.1]
    )
    check_postprocessing_prediction(pipeline, small_categorical_dataset)


@pytest.mark.parametrize(
    "metric_name, metric_evalution",
    [
        ("Statistical parity difference", (["Statistical parity difference"], [0.1])),
        ("Average odds difference", (["Average odds difference"], [0.1])),
        ("Equal opportunity difference", (["Equal opportunity difference"], [0.1])),
    ],
)
def test_reject_option_classification(
    small_categorical_dataset, metric_name, metric_evalution
):
    pipeline = build_roc_pipeline(metric_name=metric_name)
    pipeline = fit(pipeline, small_categorical_dataset)
    evaluate_pipeline(pipeline, small_categorical_dataset, *metric_evalution)
