import warnings

import numpy as np
import pandas as pd
from holisticai.datasets import Dataset

warnings.filterwarnings("ignore")


class LIMEImportanceCalculator:
    importance_type: str = "local"

    def __init__(self, learning_task, x_train, model):
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError("LIME is not installed. Please install it using 'pip install lime'") from None

        if learning_task == "regression":
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                x_train.values,
                feature_names=x_train.columns,
                discretize_continuous=True,
                mode="regression",
                random_state=42,
            )
            self.predict_function = model.predict
            self.feature_names = list(x_train.columns)
        elif learning_task in ["binary_classification", "multi_classification"]:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                x_train.values,
                feature_names=x_train.columns,
                class_names=model.classes_,
                discretize_continuous=True,
                mode="classification",
                random_state=42,
            )
            self.predict_function = model.predict_proba
            self.feature_names = list(x_train.columns)
        else:
            raise ValueError("Learning task must be regression or classification")

    def __call__(self, ds: Dataset):
        importances = []
        for i in range(len(ds)):
            instance = ds["X"].iloc[i].values.reshape(1, -1)
            exp = self.explainer.explain_instance(
                instance[0], self.predict_function, num_features=len(self.feature_names)
            )
            exp_values = np.array(next(iter(exp.local_exp.values())))
            ranked_imp = exp_values[:, 1]
            rank = exp_values[:, 0].argsort()
            importance = np.zeros_like(ranked_imp)
            importance[rank] = ranked_imp
            importances.append(importance)
        importances = np.stack(importances)
        return pd.DataFrame(data=np.abs(importances), columns=ds["X"].columns)
