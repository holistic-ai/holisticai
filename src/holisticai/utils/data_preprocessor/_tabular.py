import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def simple_preprocessor(
    X_train, X_test, y_train, y_test, categorical_types=None, only_numerics=False, model_type="classification"
):
    if categorical_types is None:
        categorical_types = ["category"]
    numeric_features = X_train.select_dtypes(exclude=categorical_types).columns.tolist()
    categorical_features = X_test.select_dtypes(include=categorical_types).columns.tolist()

    transformers = [("num", StandardScaler(), numeric_features)]
    if not only_numerics:
        transformers.append(("cat", OneHotEncoder(sparse_output=False), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)

    def transform_to_df(pipeline, X, numeric_features, categorical_features, preprocessor):
        transformed_data = pipeline.transform(X)
        if not only_numerics:
            ohe = preprocessor.named_transformers_["cat"]
            categorical_ohe_columns = ohe.get_feature_names_out(categorical_features).tolist()
            all_feature_names = numeric_features + categorical_ohe_columns
        else:
            all_feature_names = numeric_features

        return pd.DataFrame(transformed_data, columns=all_feature_names)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    pipeline.fit(X_train)

    Xt_train = transform_to_df(pipeline, X_train, numeric_features, categorical_features, preprocessor)
    Xt_test = transform_to_df(pipeline, X_test, numeric_features, categorical_features, preprocessor)

    if model_type == "classification":
        label_encoder = LabelEncoder()
        yt_train = label_encoder.fit_transform(y_train)
        yt_test = label_encoder.transform(y_test)
    else:
        yt_train = y_train
        yt_test = y_test
    return Xt_train, Xt_test, yt_train, yt_test
