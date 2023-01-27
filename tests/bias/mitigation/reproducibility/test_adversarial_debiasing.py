import os
import sys

sys.path.append(os.getcwd())

import pytest
from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import classification_bias_metrics
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult_v2

seed = 42
train_data, test_data = load_preprocessed_adult_v2()


def running_without_pipeline():
    from holisticai.bias.mitigation import AdversarialDebiasing

    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    inprocessing_model = AdversarialDebiasing(
        features_dim=X.shape[1],
        epochs=1,
        batch_size=32,
        hidden_size=64,
        adversary_loss_weight=0.1,
        verbose=1,
        use_debias=True,
        seed=seed,
    ).transform_estimator()

    fit_params = {"group_a": group_a, "group_b": group_b}
    inprocessing_model.fit(Xt, y, **fit_params)

    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = inprocessing_model.predict(Xt)

    df = classification_bias_metrics(group_a, group_b, y_pred, y, metric_type="both")
    return df


def running_with_pipeline():
    from holisticai.bias.mitigation import AdversarialDebiasing

    X, y, group_a, group_b = train_data

    inprocessing_model = AdversarialDebiasing(
        features_dim=X.shape[1],
        epochs=2,
        batch_size=16,
        hidden_size=64,
        adversary_loss_weight=0.1,
        verbose=1,
        use_debias=True,
        seed=seed,
    ).transform_estimator()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", inprocessing_model),
        ]
    )

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


@pytest.mark.skip(reason="pytorch not installed")
def test_reproducibility_with_and_without_pipeline():
    df1 = running_without_pipeline()
    df2 = running_with_pipeline()
    check_results(df1, df2)
