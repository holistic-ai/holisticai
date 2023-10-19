import os

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics, regression_bias_metrics
from holisticai.bias.mitigation import ExponentiatedGradientReduction
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
    small_regression_dataset,
)

seed = 42


def running_without_pipeline_classification(small_categorical_dataset):

    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="DemographicParity"
    ).transform_estimator(model)

    fit_params = {"group_a": group_a, "group_b": group_b}
    inprocessing_model.fit(Xt, y, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = inprocessing_model.predict(Xt)

    df = classification_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
        metric_type="both",
    )
    return df


def running_with_pipeline_classification(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset

    model = LogisticRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="ErrorRateParity"
    ).transform_estimator(model)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    df = classification_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
        metric_type="both",
    )
    return df


def running_without_pipeline_regression(small_regression_dataset):
    train_data, test_data = small_regression_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LinearRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="BoundedGroupLoss",
        loss="Absolute",
        min_val=-0.1,
        max_val=1.3,
        upper_bound=0.001,
    ).transform_estimator(model)

    fit_params = {"group_a": group_a, "group_b": group_b}
    inprocessing_model.fit(Xt, y, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = inprocessing_model.predict(Xt)
    print(y_pred.shape)
    print(y.shape)
    df = regression_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
        metric_type="both",
    )
    return df


def running_with_pipeline_regression(small_regression_dataset):
    train_data, test_data = small_regression_dataset

    model = LinearRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="BoundedGroupLoss",
        loss="Absolute",
        min_val=-0.1,
        max_val=1.3,
        upper_bound=0.001,
    ).transform_estimator(model)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    df = regression_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
        metric_type="both",
    )
    return df


def test_reproducibility_with_and_without_pipeline(
    small_categorical_dataset, small_regression_dataset
):
    df1 = running_without_pipeline_classification(small_categorical_dataset)
    df2 = running_with_pipeline_classification(small_categorical_dataset)
    check_results(df1, df2)

    df1 = running_without_pipeline_regression(small_regression_dataset)
    df2 = running_with_pipeline_regression(small_regression_dataset)
    check_results(df1, df2)
