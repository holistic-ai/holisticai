import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import check_results, small_categorical_dataset

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import Reweighing
from holisticai.pipeline import Pipeline
from holisticai.utils import extract_columns

warnings.filterwarnings("ignore")


def running_without_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data
    fit_params = {"group_a": group_a, "group_b": group_b}
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    prep = Reweighing()
    prep.fit(y, **fit_params)
    sample_weight = prep.estimator_params["sample_weight"]

    model = LogisticRegression()
    model.fit(Xt, y, sample_weight)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    Xt = prep.transform(Xt)
    y_pred = model.predict(Xt)
    df = classification_bias_metrics(
        group_b,
        group_a,
        y_pred,
        y,
    )
    return df


def running_with_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", Reweighing()),
            ("estimator", LogisticRegression()),
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
        group_b,
        group_a,
        y_pred,
        y,
    )

    return df


def test_reproducibility_with_and_without_pipeline(small_categorical_dataset):
    df1 = running_without_pipeline(small_categorical_dataset)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)


def test_reweighing():
    df_c = pd.read_csv("tests/data/small_test_classification.csv")
    group_a, group_b, y_pred_c, y_true_c = extract_columns(
        df_c, cols=["group_a", "group_b", "y_pred", "y_true"]
    )

    mitigation = Reweighing()

    fit_params = {"group_a": group_a, "group_b": group_b}
    mitigation = mitigation.fit(y_pred_c, **fit_params)
    sample_weight = mitigation.estimator_params["sample_weight"]
    expected_output = np.array(
        [2 / 3, 2 / 3, 2 / 3, 2, 1.5, 1.5, 0.75, 0.75, 0.75, 0.75]
    )
    assert_array_almost_equal(sample_weight, expected_output)
