import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from holisticai.explainability import Explainer


def regression_process_dataset():
    seed = np.random.seed(42)

    dataset = load_diabetes()
    X = dataset.data[:100, :]
    y = dataset.target[:100]
    feature_names = dataset.feature_names
    X = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )  # train test split
    return X_train, X_test, y_train, y_test, feature_names


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
def test_metrics_within_range(strategy):
    """Checks if metrics are within the valid range (0 to 1)"""
    X_train, X_test, y_train, y_test, _ = regression_process_dataset()
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="regression",
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
def test_metrics_with_invalid_top_k(strategy, alpha):
    """Checks if calling metrics with invalid top_k raises a Error"""
    X_train, X_test, y_train, y_test, _ = regression_process_dataset()
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="regression",
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
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.9])
def test_metrics_with_valid_alpha(strategy, alpha):
    """Checks if calling metrics with valid alpha works properly"""
    X_train, X_test, y_train, y_test, _ = regression_process_dataset()
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="regression",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)

    # Assert that the result is not None or empty
    assert metrics_df is not None
    assert not metrics_df.empty


@pytest.mark.parametrize("strategy", ["permutation", "surrogate", "lime", "shap"])
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.9])
def test_metrics_with_valid_input_data(strategy, alpha):
    """Checks if the explainer module works when input data is a numpy array or a pandas dataframe"""
    X_train, X_test, y_train, y_test, feature_names = regression_process_dataset()
    model = train_model(X_train, y_train)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="regression",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)
    assert isinstance(metrics_df, pd.DataFrame)
    assert metrics_df is not None
    assert not metrics_df.empty

    # Check if works with pandas dataframes
    X_test = pd.DataFrame(X_test, columns=feature_names)
    assert isinstance(X_test, pd.DataFrame)
    explainer = Explainer(
        based_on="feature_importance",
        strategy_type=strategy,
        model_type="regression",
        model=model,
        x=X_test,
        y=y_test,
    )
    metrics_df = explainer.metrics(alpha)
    assert isinstance(metrics_df, pd.DataFrame)
    assert metrics_df is not None
    assert not metrics_df.empty
