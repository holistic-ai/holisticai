import warnings

warnings.simplefilter("ignore")

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import PrejudiceRemover
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

    inprocessing_model = PrejudiceRemover(
        maxiter=100, fit_intercept=True, verbose=1, print_interval=1
    ).transform_estimator()

    fit_params = {
        "group_a": group_a,
        "group_b": group_b,
    }
    inprocessing_model.fit(Xt, y, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    predict_params = {
        "group_a": group_a,
        "group_b": group_b,
    }
    y_pred = inprocessing_model.predict(Xt, **predict_params)

    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    edf = metrics_dataframe(y_pred, y)
    return df, edf


def running_with_pipeline():

    inprocessing_model = PrejudiceRemover(
        init_type="StandarLR",
        maxiter=100,
        fit_intercept=True,
        verbose=1,
        print_interval=1,
    ).transform_estimator()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    edf = metrics_dataframe(y_pred, y)
    return df, edf


def test_reproducibility_with_and_without_pipeline():
    np.random.seed(10)
    df1, edf1 = running_without_pipeline()
    np.random.seed(10)
    df2, edf2 = running_with_pipeline()
    # df = pd.concat([edf1,edf2], axis=1)
    # df.columns = ['without pipeline' , 'with pipeline']
    # print(df)
    check_results(df1, df2)


# test_reproducibility_with_and_without_pipeline()
