import os
import sys

sys.path.append(os.getcwd())

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import regression_bias_metrics
from holisticai.bias.mitigation import PluginEstimationAndCalibration
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime
from tests.testing_utils._tests_utils import check_results

seed = 42
train_data, test_data = load_preprocessed_us_crime()


def running_without_pipeline():

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


def running_with_pipeline():
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


def test_reproducibility_with_and_without_pipeline():
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline()
    np.random.seed(seed)
    df2 = running_with_pipeline()
    check_results(df1, df2)
