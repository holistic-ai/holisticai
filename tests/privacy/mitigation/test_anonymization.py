import numpy as np
import pandas as pd

from holisticai.privacy.mitigation import Anonymize


def prepare_data():
    from sklearn.model_selection import train_test_split

    from holisticai.datasets import load_dataset

    loaded = load_dataset(dataset="adult", preprocessed=False, as_array=False)
    df = pd.DataFrame(data=loaded.data, columns=loaded.feature_names)
    df["class"] = loaded.target.apply(lambda x: 1 if x == ">50K" else 0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.set_index(pd.Series(range(len(X_train))))
    categorical_features = X.select_dtypes(include=["category"]).columns
    features = X_train.columns
    return X_train, y_train, features, categorical_features


def test_anonymization():
    QI = ["education", "marital-status", "age"]
    X_train, y_train, features, categorical_features = prepare_data()
    anonymizer = Anonymize(
        100,
        QI,
        categorical_features=list(categorical_features),
        features_names=features,
    )
    # Test if the function runs successfully without errors
    try:
        anon = anonymizer.anonymize(X_train, y_train)
        assert True, "Function ran successfully"
    except Exception as e:
        assert False, f"Function encountered an error: {e}"
    return anon
