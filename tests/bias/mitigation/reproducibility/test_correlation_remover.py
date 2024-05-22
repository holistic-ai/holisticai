import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.preprocessing import StandardScaler

from holisticai.mitigation.bias import CorrelationRemover
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_regression_dataset,
)

seed = 42


def running_without_pipeline(small_regression_dataset):
    from sklearn.linear_model import LinearRegression

    from holisticai.metrics.bias import regression_bias_metrics

    train_data, test_data = small_regression_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    prep = CorrelationRemover()

    fit_params = {"group_a": group_a, "group_b": group_b}

    Xt = prep.fit_transform(Xt, **fit_params)

    model = LinearRegression()

    model.fit(Xt, y)

    # Test
    X, y, group_a, group_b = test_data
    transform_params = {"group_a": group_a, "group_b": group_b}
    Xt = scaler.transform(X)
    Xt = prep.transform(Xt, **transform_params)

    y_pred = model.predict(Xt)

    df = regression_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
        metric_type="both",
    )
    return df


def running_with_pipeline(small_regression_dataset):
    from sklearn.linear_model import LinearRegression

    from holisticai.metrics.bias import regression_bias_metrics

    train_data, test_data = small_regression_dataset
    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", CorrelationRemover()),
            ("model", model),
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


def test_reproducibility_with_and_without_pipeline(small_regression_dataset):
    df1 = running_without_pipeline(small_regression_dataset)
    df2 = running_with_pipeline(small_regression_dataset)
    check_results(df1, df2)
