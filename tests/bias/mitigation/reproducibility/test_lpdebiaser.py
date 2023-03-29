import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import (
    check_results,
    small_categorical_dataset,
    small_multiclass_dataset,
)

from holisticai.bias.metrics import classification_bias_metrics, multiclass_bias_metrics
from holisticai.bias.mitigation import LPDebiaserBinary, LPDebiaserMulticlass
from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import SensitiveGroups

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xt, y)

    y_proba = model.predict_proba(Xt)

    fit_params = {
        "group_a": group_a,
        "group_b": group_b,
    }

    post = LPDebiaserBinary()
    post.fit(y_true=y, y_proba=y_proba, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_proba = model.predict_proba(Xt)
    y_pred = post.transform(y_proba=y_proba, **transform_params)["y_pred"]

    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    return df


def running_with_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", LPDebiaserBinary()),
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
    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    return df


def running_without_pipeline_multiclass(small_multiclass_dataset):
    train_data, test_data = small_multiclass_dataset

    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xt, y)

    y_pred = model.predict(Xt)

    fit_params = {"group_a": group_a, "group_b": group_b}
    post = LPDebiaserMulticlass(constraint="EqualizedOpportunity")
    post.fit(y, y_pred, **fit_params)

    X, y, group_a, group_b = test_data
    sens = SensitiveGroups().fit_transform(
        np.stack([group_a, group_b], axis=1), convert_numeric=True
    )

    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_pred = model.predict(Xt)
    y_pred = post.transform(y_pred, **transform_params)["y_pred"]

    df = multiclass_bias_metrics(sens, y_pred, y, metric_type="both")
    return df


def running_with_pipeline_multiclass(small_multiclass_dataset):
    train_data, test_data = small_multiclass_dataset

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            (
                "bm_posprocessing",
                LPDebiaserMulticlass(constraint="EqualizedOpportunity"),
            ),
        ]
    )

    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    sens = SensitiveGroups().fit_transform(
        np.stack([group_a, group_b], axis=1), convert_numeric=True
    )

    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)

    df = multiclass_bias_metrics(sens, y_pred, y, metric_type="both")
    return df


def test_reproducibility_with_and_without_pipeline_multiclass(small_multiclass_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline_multiclass(small_multiclass_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline_multiclass(small_multiclass_dataset)
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_binary(small_categorical_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline(small_categorical_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)
