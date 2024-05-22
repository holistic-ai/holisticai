"""
This module provides functions for computing surrogate feature importance and efficacy metrics.
"""

import pandas as pd

from holisticai.explainability.plots import DecisionTreeVisualizer

from ..global_importance import (
    ExplainabilityEase,
    FourthFifths,
    SpreadDivergence,
    SpreadRatio,
    SurrogacyMetric,
)
from ..utils import (
    BaseFeatureImportance,
    GlobalFeatureImportance,
    check_feature_importance,
)


def create_surrogate_model(model_type, x, y):
    """
    Create a surrogate model for a given model type, input features and predicted output.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        x (pandas.DataFrame): The input features.
        y (numpy.ndarray): The predicted output.

    Returns:
        sklearn.tree.DecisionTreeClassifier or sklearn.tree.DecisionTreeRegressor: The surrogate model.
    """
    if model_type == "binary_classification":
        from sklearn.tree import DecisionTreeClassifier

        dt = DecisionTreeClassifier(max_depth=3)
        return dt.fit(x, y)
    elif model_type == "regression":
        from sklearn.tree import DecisionTreeRegressor

        dt = DecisionTreeRegressor(max_depth=3)
        return dt.fit(x, y)
    else:
        raise ValueError(
            "model_type must be either 'binary_classification' or 'regression'"
        )


def compute_surrogate_feature_importance(model_type, x, y_pred):
    """
    Compute surrogate feature importance for a given model type, model and input features.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        model (sklearn estimator): The model to compute surrogate feature importance for.
        x (pandas.DataFrame): The input features.

    Returns:
        holisticai.explainability.feature_importance.SurrogateFeatureImportance: The surrogate feature importance.
    """
    surrogate = create_surrogate_model(model_type, x, y_pred)
    feature_names = x.columns
    forest = surrogate

    sorted_features = sorted(
        zip(feature_names, forest.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    features_dict = dict(sorted_features)
    df_feat_imp = pd.DataFrame(
        {"Variable": features_dict.keys(), "Importance": features_dict.values()}
    )

    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp["Importance"] /= df_feat_imp["Importance"].sum()
    df_feat_imp = df_feat_imp.sort_values("Importance", ascending=False).copy()

    return SurrogateFeatureImportance(model_type, x, y_pred, df_feat_imp, surrogate)


class SurrogateFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(self, model_type, x, y_pred, importance_weights, surrogate):
        self.model_type = model_type
        self.y = y_pred
        self.x = x
        self.feature_importance = importance_weights
        self.surrogate = self.model = surrogate
        self.tree_visualizer = DecisionTreeVisualizer()

    def metrics(self, alpha, detailed, metric_names=None):
        if metric_names is None:
            metric_names = [
                "Explainability Ease",
                "Fourth Fifths",
                "Spread Divergence",
                "Spread Ratio",
                "Surrogacy Efficacy",
            ]

        (feat_imp, _), (alpha_feat_imp, _) = self.get_alpha_feature_importance(alpha)

        if len(alpha_feat_imp) == 0:
            print(
                f"There are no features for alpha={alpha}, please select a higher value."
            )
            return None

        sd = SpreadDivergence(detailed=detailed)
        sr = SpreadRatio(detailed=detailed)
        ff = FourthFifths(detailed=detailed)
        expe = ExplainabilityEase(
            model_type=self.model_type, model=self.surrogate, x=self.x
        )
        sur_eff = SurrogacyMetric(model_type=self.model_type)

        metric_scores = []

        score = sd(alpha_feat_imp)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Reference": sd.reference}
            for metric_name, value in score.items()
        ]

        score = sr(alpha_feat_imp)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Reference": sr.reference}
            for metric_name, value in score.items()
        ]

        score = ff(feat_imp)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Reference": ff.reference}
            for metric_name, value in score.items()
        ]

        score = expe(alpha_feat_imp)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Reference": expe.reference}
            for metric_name, value in score.items()
        ]

        score = sur_eff(self.surrogate, self.x, self.y)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Reference": sur_eff.reference}
            for metric_name, value in score.items()
        ]

        return pd.DataFrame(metric_scores).set_index("Metric").sort_index()

    def tree_visualization(self, backend="sklearn", **kargs):
        if backend in self.tree_visualizer.visualization_backend:
            return self.tree_visualizer.show(backend, self, **kargs)
        else:
            available_packages = ", ".join(
                list(self.tree_visualizer.visualization_backend.keys())
            )
            raise Exception(
                f"Unknown backend. Available backends are: {available_packages}"
            )
