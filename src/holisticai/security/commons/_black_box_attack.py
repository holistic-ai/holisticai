import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def append_if_not_empty(original_array, array_to_append):
    if len(array_to_append) > 0:
        original_array = np.append(original_array, array_to_append)
    return original_array


class BlackBoxAttack:
    def __init__(self, attacker_estimator, attack_feature, attack_train_ratio=0.5):
        self.attack_feature = attack_feature
        self.attack_train_ratio = attack_train_ratio
        self.attacker_estimator = attacker_estimator

    def create_preprocessor(self, X):
        categorical_features = X.select_dtypes(include=["category"]).columns
        numerical_fatures = X.select_dtypes(exclude=["category"]).columns

        # Create transformers for numerical and categorical features
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

        # Combine transformers into a preprocessor using ColumnTransformer
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_fatures),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def fit(self, X, y):
        """

        Description
        -----------
        The black-box attack basically trains an additional classifier (called the attack model) to predict the attacked feature's value from the remaining n-1
        features as well as the original (attacked) model's predictions.

        Parameters
        ----------

        X : pandas Dataframe
            input matrix

        y : numpy array
            Target vector of original model

        Returns
        -------

        np.ndarray: Predicted output per sample.
        """

        categorical_features = []
        y_train_attack = X[self.attack_feature]
        X_train_attack = X.drop(columns=[self.attack_feature])
        X_train_attack["label"] = y
        if y_train_attack.dtype == "category":
            categorical_features.append("label")
        # Create transformers for numerical and categorical features

        preprocessor = self.create_preprocessor(X_train_attack)

        attacker = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", self.attacker_estimator)])

        attack_train_size = int(X_train_attack.shape[0] * self.attack_train_ratio)
        attacker.fit(X_train_attack[:attack_train_size], y_train_attack[:attack_train_size])
        self.attacker = attacker
        return self

    def transform(self, X, y):
        y_attack = X[self.attack_feature]
        X_test_attack = X.drop(columns=[self.attack_feature])
        X_test_attack["label"] = y
        y_pred_attack = self._predict(X_test_attack)
        return y_attack, y_pred_attack

    def _predict(self, x_attack):
        return self.attacker.predict(x_attack)


def classification_security_features(X, y, attacker, attack_feature):
    if attacker == "black_box":
        attacker = BlackBoxAttack(learning_task="classification", attack_feature=attack_feature, attack_train_ratio=0.5)
        attacker.fit(X, y)
        return attacker
    raise ValueError("Invalid attacker type. Please choose from 'black_box'")
