import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import (
    check_results,
    small_categorical_dataset,
    small_multiclass_dataset,
)

from holisticai.bias.metrics import classification_bias_metrics, multiclass_bias_metrics
from holisticai.bias.mitigation import MLDebiaser
from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import SensitiveGroups

seed = 42


def running_without_pipeline_multiclass(small_multiclass_dataset):
    train_data, test_data = small_multiclass_dataset

    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xt, y)

    y_proba = model.predict_proba(Xt)

    fit_params = {"group_a": group_a, "group_b": group_b}
    post = MLDebiaser(max_iter=2)
    post.fit(y_proba, **fit_params)

    X, y, group_a, group_b = test_data
    sens = SensitiveGroups().fit_transform(
        np.stack([group_a, group_b], axis=1), convert_numeric=True
    )

    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_proba = model.predict_proba(Xt)
    y_pred = post.transform(y_proba, **transform_params)["y_pred"]

    df = multiclass_bias_metrics(sens, y_pred, y, metric_type="both")
    return df


def running_with_pipeline_multiclass(small_multiclass_dataset):
    train_data, test_data = small_multiclass_dataset

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", MLDebiaser(max_iter=2)),
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


def running_without_pipeline_binary(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset

    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xt, y)

    y_proba = model.predict_proba(Xt)

    fit_params = {"group_a": group_a, "group_b": group_b}
    post = MLDebiaser(max_iter=2)
    post.fit(y_proba, **fit_params)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_proba = model.predict_proba(Xt)
    y_pred = post.transform(y_proba, **transform_params)["y_pred"]

    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    return df


def running_with_pipeline_binary(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", MLDebiaser(max_iter=2)),
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


def test_reproducibility_with_and_without_pipeline_multiclass(small_multiclass_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline_multiclass(small_multiclass_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline_multiclass(small_multiclass_dataset)
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_binary(small_categorical_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline_binary(small_categorical_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline_binary(small_categorical_dataset)
    check_results(df1, df2)
