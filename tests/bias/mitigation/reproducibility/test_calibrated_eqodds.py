import os
import sys

sys.path.append(os.getcwd())
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.mitigation import CalibratedEqualizedOdds
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

warnings.filterwarnings("ignore")

seed = 42
train_data, test_data = load_preprocessed_adult_v2()


def running_without_pipeline():

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

    post = CalibratedEqualizedOdds("fpr")
    post.fit(y, y_proba, **fit_params)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    transform_params = {"group_a": group_a, "group_b": group_b}

    y_pred = model.predict(Xt)
    y_proba = model.predict_proba(Xt)
    y_pred = post.transform(y_pred, y_proba, **transform_params)["y_pred"]

    df = classification_bias_metrics(
        group_a.to_numpy().ravel(),
        group_b.to_numpy().ravel(),
        y_pred,
        y.to_numpy().ravel(),
    )
    return df


def running_with_pipeline():
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", CalibratedEqualizedOdds("fpr")),
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
        group_a.to_numpy().ravel(),
        group_b.to_numpy().ravel(),
        y_pred,
        y.to_numpy().ravel(),
    )
    return df


def test_reproducibility_with_and_without_pipeline():
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline()
    np.random.seed(seed)
    df2 = running_with_pipeline()
    check_results(df1, df2)
