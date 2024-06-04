import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.metrics.bias import classification_bias_metrics
from holisticai.mitigation.bias import RejectOptionClassification
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
)

seed = 42


def running_without_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xt, y)

    y_proba = model.predict_proba(Xt)

    fit_params = {"group_a": group_a, "group_b": group_b}
    post = RejectOptionClassification()
    post.fit(y, y_proba, **fit_params)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_pred = model.predict(Xt)
    y_proba = model.predict_proba(Xt)

    y_pred = post.transform(y_pred, y_proba, **transform_params)["y_pred"]

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
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", RejectOptionClassification()),
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
    np.random.seed(seed)
    df1 = running_without_pipeline(small_categorical_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)
