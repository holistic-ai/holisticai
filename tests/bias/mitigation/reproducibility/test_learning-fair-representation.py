import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import LearningFairRepresentation
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
)

warnings.filterwarnings("ignore")
seed = 42


def running_without_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data
    fit_params = {"group_a": group_a, "group_b": group_b}
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    prep = LearningFairRepresentation(k=10, Ax=0.1, Ay=1.0, Az=2.0, seed=seed)
    Xt = prep.fit_transform(Xt, y, **fit_params)

    model = LogisticRegression()
    model.fit(Xt, y)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {
        "group_a": group_a,
        "group_b": group_b,
    }
    Xt = prep.transform(Xt, **transform_params)
    y_pred = model.predict(Xt)
    df = classification_bias_metrics(group_b, group_a, y_pred, y, metric_type="both")
    return df


def running_with_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "bm_preprocessing",
                LearningFairRepresentation(k=10, Ax=0.1, Ay=1.0, Az=2.0, seed=seed),
            ),
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
    df = classification_bias_metrics(group_b, group_a, y_pred, y, metric_type="both")
    return df


def test_reproducibility_with_and_without_pipeline(small_categorical_dataset):
    df1 = running_without_pipeline(small_categorical_dataset)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)
