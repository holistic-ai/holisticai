import os

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.metrics.bias import classification_bias_metrics, regression_bias_metrics
from holisticai.mitigation.bias import ExponentiatedGradientReduction
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
    small_regression_dataset,
)

seed = 42


def running_without_pipeline_classification(small_categorical_dataset):

    train = small_categorical_dataset['train']
    test = small_categorical_dataset['test']

    scaler = StandardScaler()
    Xt = scaler.fit_transform(train['x'])

    model = LogisticRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="DemographicParity"
    ).transform_estimator(model)

    fit_params = {"group_a": train['group_a'], "group_b": train['group_b']}
    inprocessing_model.fit(Xt, train['y'], **fit_params)

    # Test
    Xt = scaler.transform(test['x'])

    y_pred = inprocessing_model.predict(Xt)

    df = classification_bias_metrics(
        test['group_a'],
        test['group_b'],
        y_pred,
        test['y'],
        metric_type="both",
    )
    return df


def running_with_pipeline_classification(small_categorical_dataset):
    train = small_categorical_dataset['train']
    test = small_categorical_dataset['test']

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

    fit_params = {"bm__group_a": train['group_a'], "bm__group_b": train['group_b']}

    pipeline.fit(train['x'], train['y'], **fit_params)

    predict_params = {
        "bm__group_a": test['group_a'],
        "bm__group_b": test['group_b'],
    }
    y_pred = pipeline.predict(test['x'], **predict_params)
    df = classification_bias_metrics(
        test['group_a'],
        test['group_b'],
        y_pred,
        test['y'],
        metric_type="both",
    )
    return df


def running_without_pipeline_regression(small_regression_dataset):
    train = small_regression_dataset['train']
    test = small_regression_dataset['test']
    

    scaler = StandardScaler()
    Xt = scaler.fit_transform(train['x'])

    model = LinearRegression()
    inprocessing_model = ExponentiatedGradientReduction(
        constraints="BoundedGroupLoss",
        loss="Absolute",
        min_val=-0.1,
        max_val=1.3,
        upper_bound=0.001,
    ).transform_estimator(model)

    fit_params = {"group_a": train['group_a'], "group_b": train['group_b']}
    inprocessing_model.fit(Xt, train['y'], **fit_params)

    # Test
    Xt = scaler.transform(test['x'])

    y_pred = inprocessing_model.predict(Xt)

    df = regression_bias_metrics(
        test['group_a'],
        test['group_b'],
        y_pred,
        test['y'],
        metric_type="both",
    )
    return df


def running_with_pipeline_regression(small_regression_dataset):
    train = small_regression_dataset['train']
    test = small_regression_dataset['test']

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

    fit_params = {"bm__group_a": train['group_a'], "bm__group_b": train['group_b']}

    pipeline.fit(train['x'], train['y'], **fit_params)

    predict_params = {
        "bm__group_a": test['group_a'],
        "bm__group_b": test['group_b'],
    }
    y_pred = pipeline.predict(test['x'], **predict_params)
    df = regression_bias_metrics(
        test['group_a'],
        test['group_b'],
        y_pred,
        test['y'],
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
