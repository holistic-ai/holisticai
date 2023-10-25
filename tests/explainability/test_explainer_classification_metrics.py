import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from holisticai.explainability import Explainer
from tests.bias.mitigation.testing_utils.utils import (
     small_categorical_dataset,
)


def classification_process_dataset(small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X_train, y_train, _, _ = train_data
    X_test, y_test, _, _ = test_data
    return X_train, X_test, y_train, y_test, None


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
def test_metrics_within_range(small_categorical_dataset, strategy):
    """Checks if metrics are within the valid range (0 to 1)"""
    X_train, X_test, y_train, y_test, _ = classification_process_dataset(small_categorical_dataset)
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="binary_classification",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics = explainer.metrics()

    # Define the range for valid metric values (0 to 1)
    min_valid_value = 0
    max_valid_value = 1

    not_ranged_metrics = ["Importance Spread Divergence"]

    for index, row in metrics.iterrows():
        if index in not_ranged_metrics:
            continue
        assert (
            min_valid_value <= row["Value"] <= max_valid_value
        ), f"Metric {index} has an invalid value: {row['Value']}"


@pytest.mark.xfail(reason="Expected to raise DivisionByZero or InvalidComparison")
@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
@pytest.mark.parametrize("alpha", [-1, 0, "1"])
def test_metrics_with_invalid_top_k(small_categorical_dataset, strategy, alpha):
    """Checks if calling metrics with invalid top_k raises a Error"""
    X_train, X_test, y_train, y_test, _ = classification_process_dataset(small_categorical_dataset)
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="binary_classification",
        model=model,
        x=X_test,
        y=y_test,
    )
    try:
        explainer.metrics(alpha)
    except ZeroDivisionError:
        pytest.xfail("ZeroDivisionError raised")
    except TypeError:
        pytest.xfail("TypeError raised")


@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_metrics_with_valid_alpha(small_categorical_dataset, strategy, alpha):
    """Checks if calling metrics with valid alpha works properly"""
    X_train, X_test, y_train, y_test, _ = classification_process_dataset(small_categorical_dataset)
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="binary_classification",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)

    # Assert that the result is not None or empty
    assert metrics_df is not None
    assert not metrics_df.empty


@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_metrics_with_valid_input_data(small_categorical_dataset, strategy, alpha):
    """Checks if the explainer module works when input data is a numpy array or a pandas dataframe"""
    X_train, X_test, y_train, y_test, _ = classification_process_dataset(small_categorical_dataset)
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="binary_classification",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)
    assert isinstance(metrics_df, pd.DataFrame)
    assert metrics_df is not None
    assert not metrics_df.empty

    # Check if works with pandas dataframes
    X_test = pd.DataFrame(X_test)
    assert isinstance(X_test, pd.DataFrame)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="binary_classification",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)
    assert isinstance(metrics_df, pd.DataFrame)
    assert metrics_df is not None
    assert not metrics_df.empty
