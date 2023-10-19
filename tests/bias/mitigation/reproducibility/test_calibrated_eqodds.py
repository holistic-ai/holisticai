import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import CalibratedEqualizedOdds
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
)

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset

    X_train, y_train, group_a_train, group_b_train = train_data

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_train_proba = model.predict_proba(X_train_scaled)

    post = CalibratedEqualizedOdds("fpr")
    post.fit(y_train, y_train_proba, group_a=group_a_train, group_b=group_b_train)

    # Test
    X_test, y_test, group_a_test, group_b_test = test_data
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    y_pred = post.transform(
        y_pred, y_test_proba, group_a=group_a_test, group_b=group_b_test
    )["y_pred"]

    df = classification_bias_metrics(group_a_test, group_b_test, y_pred, y_test)
    return df


def running_with_pipeline(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", CalibratedEqualizedOdds("fpr")),
        ]
    )

    X_train, y_train, group_a_train, group_b_train = train_data
    pipeline.fit(X_train, y_train, bm__group_a=group_a_train, bm__group_b=group_b_train)

    X_test, y_test, group_a_test, group_b_test = test_data
    y_pred = pipeline.predict(
        X_test, bm__group_a=group_a_test, bm__group_b=group_b_test
    )

    df = classification_bias_metrics(group_a_test, group_b_test, y_pred, y_test)
    return df


def test_reproducibility_with_and_without_pipeline(small_categorical_dataset):
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline(small_categorical_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)
