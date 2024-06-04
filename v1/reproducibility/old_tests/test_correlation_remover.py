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

    train = small_regression_dataset['train']
    test = small_regression_dataset['test']

    scaler = StandardScaler()
    print(train['x'])
    Xt = scaler.fit_transform(train['x'])

    prep = CorrelationRemover()

    fit_params = {"group_a": train['group_a'], "group_b": train['group_b']}

    Xt = prep.fit_transform(Xt, **fit_params)

    model = LinearRegression()

    model.fit(Xt, train['y'])

    # Test
    transform_params = {"group_a": test['group_a'], "group_b": test['group_b']}
    Xt = scaler.transform(test['x'])
    Xt = prep.transform(Xt, **transform_params)

    y_pred = model.predict(Xt)

    df = regression_bias_metrics(
        test['group_a'],
        test['group_b'],
        y_pred,
        test['y'],
        metric_type="both",
    )
    return df


def running_with_pipeline(small_regression_dataset):
    from sklearn.linear_model import LinearRegression

    from holisticai.metrics.bias import regression_bias_metrics

    train = small_regression_dataset['train']
    test = small_regression_dataset['test']

    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", CorrelationRemover()),
            ("model", model),
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


def test_reproducibility_with_and_without_pipeline(small_regression_dataset):
    df1 = running_without_pipeline(small_regression_dataset)
    df2 = running_with_pipeline(small_regression_dataset)
    check_results(df1, df2)
