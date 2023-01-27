import os
import sys

sys.path.append(os.getcwd())

from sklearn.preprocessing import StandardScaler

from holisticai.bias.mitigation import CorrelationRemover
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

seed = 42


def running_without_pipeline():
    from sklearn.linear_model import LinearRegression

    from holisticai.bias.metrics import regression_bias_metrics

    train_data, test_data = load_preprocessed_us_crime()
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


def running_with_pipeline():
    from sklearn.linear_model import LinearRegression

    from holisticai.bias.metrics import regression_bias_metrics

    train_data, test_data = load_preprocessed_us_crime()
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


def test_reproducibility_with_and_without_pipeline():
    df1 = running_without_pipeline()
    df2 = running_with_pipeline()
    check_results(df1, df2)


# test_reproducibility_with_and_without_pipeline()
