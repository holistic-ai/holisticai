import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "tests"))

import numpy as np
from sklearn.preprocessing import StandardScaler

from holisticai.bias.mitigation import ExponentiatedGradientReduction
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

seed = 42


def running_without_pipeline_classification():
    from sklearn.linear_model import LogisticRegression

    from holisticai.bias.metrics import classification_bias_metrics

    train_data, test_data = load_preprocessed_adult_v2()
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


def running_with_pipeline_classification():
    from sklearn.linear_model import LogisticRegression

    from holisticai.bias.metrics import classification_bias_metrics

    train_data, test_data = load_preprocessed_adult_v2()

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


def running_without_pipeline_regression():
    from sklearn.linear_model import LinearRegression

    from holisticai.bias.metrics import regression_bias_metrics

    train_data, test_data = load_preprocessed_us_crime()
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


def running_with_pipeline_regression():
    from sklearn.linear_model import LinearRegression

    from holisticai.bias.metrics import regression_bias_metrics

    train_data, test_data = load_preprocessed_us_crime()

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


def test_reproducibility_with_and_without_pipeline():
    df1 = running_without_pipeline_classification()
    df2 = running_with_pipeline_classification()
    check_results(df1, df2)

    df1 = running_without_pipeline_regression()
    df2 = running_with_pipeline_regression()
    check_results(df1, df2)


test_reproducibility_with_and_without_pipeline()
