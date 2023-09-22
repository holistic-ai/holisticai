import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import check_results, small_regression_dataset

from holisticai.bias.metrics import regression_bias_metrics
from holisticai.bias.mitigation import PluginEstimationAndCalibration
from holisticai.pipeline import Pipeline

seed = 42


def running_without_pipeline(small_regression_dataset):
    train_data, test_data = small_regression_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xt, y)

    y_pred = model.predict(Xt)

    fit_params = {"group_a": group_a, "group_b": group_b}
    post = PluginEstimationAndCalibration()
    post.fit(y_pred, **fit_params)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_pred = model.predict(Xt)
    y_pred = post.transform(y_pred, **transform_params)["y_pred"]

    df = regression_bias_metrics(
        group_a,
        group_b,
        y_pred,
        y,
    )
    return df


def running_with_pipeline(small_regression_dataset):
    train_data, test_data = small_regression_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LinearRegression()),
            ("bm_posprocessing", PluginEstimationAndCalibration()),
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
    )
    return df


def test_reproducibility_with_and_without_pipeline(small_regression_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline(small_regression_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_regression_dataset)
    check_results(df1, df2)
