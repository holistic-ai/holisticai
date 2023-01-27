import os
import sys

sys.path = ["./"] + sys.path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics, multiclass_bias_metrics
from holisticai.bias.mitigation import MLDebiaser
from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import SensitiveGroups
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

seed = 42


def running_without_pipeline_multiclass():
    nb_classes = 5
    train_data, test_data = load_preprocessed_us_crime(nb_classes=nb_classes)

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


def running_with_pipeline_multiclass():
    nb_classes = 5
    train_data, test_data = load_preprocessed_us_crime(nb_classes=nb_classes)

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


def running_without_pipeline_binary():
    train_data, test_data = load_preprocessed_adult_v2()

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


def running_with_pipeline_binary():
    train_data, test_data = load_preprocessed_adult_v2()

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


def test_reproducibility_with_and_without_pipeline_multiclass():
    np.random.seed(seed)
    df1 = running_without_pipeline_multiclass()
    np.random.seed(seed)
    df2 = running_with_pipeline_multiclass()
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_binary():
    np.random.seed(seed)
    df1 = running_without_pipeline_binary()
    np.random.seed(seed)
    df2 = running_with_pipeline_binary()
    check_results(df1, df2)


test_reproducibility_with_and_without_pipeline_multiclass()
test_reproducibility_with_and_without_pipeline_binary()
