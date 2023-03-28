import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import MetaFairClassifier
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_utils import (
    check_results,
    load_preprocessed_adult,
    metrics_dataframe,
)

seed = 42
train_data, test_data = load_preprocessed_adult()


def running_without_pipeline():
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    inprocessing_model = MetaFairClassifier(
        constraint="StatisticalRate", verbose=1
    ).transform_estimator()

    fit_params = {"group_a": group_a, "group_b": group_b}
    inprocessing_model.fit(Xt, y, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = inprocessing_model.predict(Xt)

    df = classification_bias_metrics(group_b, group_a, y_pred, y, metric_type="both")
    edf = metrics_dataframe(y_pred, y)
    return df, edf


def running_with_pipeline():
    inprocessing_model = MetaFairClassifier(
        constraint="FalseDiscovery", verbose=1
    ).transform_estimator()

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
    df = classification_bias_metrics(group_b, group_a, y_pred, y, metric_type="both")
    edf = metrics_dataframe(y_pred, y)
    return df, edf


def test_reproducibility_with_and_without_pipeline():
    df1, edf1 = running_without_pipeline()
    df2, edf2 = running_with_pipeline()
    df = pd.concat([edf1, edf2], axis=1)
    df.columns = ["without pipeline", "with pipeline"]
    check_results(df1, df2)


# test_reproducibility_with_and_without_pipeline()
