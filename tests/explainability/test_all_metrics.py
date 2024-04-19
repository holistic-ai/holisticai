import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from holisticai.datasets import load_dataset
from holisticai.explainability.metrics.core import (
    explainability_ease,
    position_parity,
    rank_alignment,
)
from holisticai.explainability.metrics.global_importance._explainability_level import (
    compute_partial_dependence,
)
from holisticai.explainability.metrics.utils import get_index_groups


def convert_float_to_categorical(target, nb_classes, numeric_classes=True):
    eps = np.finfo(float).eps
    if numeric_classes:
        labels = list(range(nb_classes))
    else:
        labels = [f"Q{c}-Q{c+1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for i, c in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype(np.int32)


def get_feat_importance(x, y, model, samples_len):
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
    return df_feat_imp.sort_values("Importance", ascending=False).copy()


def binary_classification_process_dataset():
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


def multiclass_classification_process_dataset():
    seed = np.random.seed(42)
    df, _, _ = load_dataset(dataset="crime", preprocessed=True, as_array=False)

    nb_classes = 5
    X = df.iloc[:, :-1]
    y = convert_float_to_categorical(df.iloc[:, -1], nb_classes=nb_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )  # train test split
    return X_train, X_test, y_train, y_test


def test_binary_classification_position_parity():
    X_train, X_test, y_train, _, _ = binary_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="binary_classification", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert position_parity(feat_importance, conditional_feature_importance) is not None


def test_regression_position_parity():
    X_train, X_test, y_train, _, _ = regression_process_dataset()
    model = train_model_regression(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="regression", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert position_parity(feat_importance, conditional_feature_importance) is not None


def test_multiclass_classification_position_parity():
    X_train, X_test, y_train, _ = multiclass_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="multiclass_classification", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert position_parity(feat_importance, conditional_feature_importance) is not None


def test_binary_classification_rank_alignment():
    X_train, X_test, y_train, _, _ = binary_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="binary_classification", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert rank_alignment(feat_importance, conditional_feature_importance) is not None


def test_regression_rank_alignment():
    X_train, X_test, y_train, _, _ = regression_process_dataset()
    model = train_model_regression(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="regression", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert rank_alignment(feat_importance, conditional_feature_importance) is not None


def test_multiclass_classification_rank_alignment():
    X_train, X_test, y_train, _ = multiclass_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    y = pd.Series(pred, index=X_test.index)
    index_groups = get_index_groups(model_type="multiclass_classification", y=y)
    conditional_feature_importance = [
        get_feat_importance(
            x=X_test.loc[index], y=y.loc[index], model=model, samples_len=len(index)
        )
        for _, index in index_groups.items()
    ]
    assert rank_alignment(feat_importance, conditional_feature_importance) is not None


def test_binary_classification_explainability_ease():
    X_train, X_test, y_train, _, _ = binary_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    partial_dependence = compute_partial_dependence(
        model=model, feature_importance=feat_importance, x=X_test, target=1
    )
    assert explainability_ease(partial_dependence_list=[partial_dependence]) is not None


def test_regression_explainability_ease():
    X_train, X_test, y_train, _, _ = regression_process_dataset()
    model = train_model_regression(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    partial_dependence = compute_partial_dependence(
        model=model, feature_importance=feat_importance, x=X_test, target=None
    )
    assert explainability_ease(partial_dependence_list=[partial_dependence]) is not None


def test_multiclass_classification_explainability_ease():
    X_train, X_test, y_train, _ = multiclass_classification_process_dataset()
    model = train_model_classification(X_train, y_train)
    pred = model.predict(X_test)
    feat_importance = get_feat_importance(
        x=X_test, y=pred, model=model, samples_len=len(X_test)
    )
    partial_dependence = [
        compute_partial_dependence(
            model=model, feature_importance=feat_importance, x=X_test, target=i
        )
        for i in np.unique(pred)
    ]
    assert explainability_ease(partial_dependence_list=partial_dependence) is not None
