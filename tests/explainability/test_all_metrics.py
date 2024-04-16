import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_diabetes

from holisticai.datasets import load_dataset

from holisticai.explainability.metrics.core.all_metrics import position_parity

from holisticai.explainability.metrics.utils import (
    get_index_groups,
)


def get_feat_importance_ind(x, y, model, samples_len):
    max_samples = min(1000, samples_len)
    feat_imp = permutation_importance(model, x, y, n_jobs=-1, max_samples=max_samples)
    df_feat_imp = pd.DataFrame(
        {
            "Variable": x.columns,
            "Importance": feat_imp["importances_mean"],
        }
    )
    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp["Importance"] /= df_feat_imp["Importance"].sum()
    return df_feat_imp.sort_values("Importance", ascending=False).copy().index


def classification_process_dataset():
    df, _, _ = load_dataset(dataset="adult", preprocessed=True, as_array=False)
    X = df.iloc[:500, :-1]
    y = df.iloc[:500, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, None


def train_model_classification(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


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


def train_model_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def test_binary_classification_position_parity():
    X_train, X_test, y_train, y_test, _ = classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance_ind(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="binary_classification", y=y)
    conditional_features_importance = [
        get_feat_importance_ind(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert position_parity(feat_importance, conditional_features_importance) is not None


def test_regression_position_parity():
    X_train, X_test, y_train, y_test, _ = regression_process_dataset()
    model = train_model_regression(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance_ind(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="regression", y=y)
    conditional_features_importance = [
        get_feat_importance_ind(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert position_parity(feat_importance, conditional_features_importance) is not None
