import os
import sys

import numpy as np

sys.path = ["./"] + sys.path
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics, multiclass_bias_metrics
from holisticai.bias.mitigation import LPDebiaserBinary, LPDebiaserMulticlass
from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import SensitiveGroups
from tests.testing_utils._tests_data_utils import load_preprocessed_us_crime
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline():
    train_data, test_data = load_preprocessed_adult_v2()
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


def running_with_pipeline():
    train_data, test_data = load_preprocessed_adult_v2()

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


def running_without_pipeline_multiclass():
    nb_classes = 5
    train_data, test_data = load_preprocessed_us_crime(nb_classes=nb_classes)

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


def running_with_pipeline_multiclass():
    nb_classes = 5
    train_data, test_data = load_preprocessed_us_crime(nb_classes=nb_classes)

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


def test_reproducibility_with_and_without_pipeline_multiclass():
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline()
    np.random.seed(seed)
    df2 = running_with_pipeline()
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_binary():
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline()
    np.random.seed(seed)
    df2 = running_with_pipeline()
    check_results(df1, df2)
