from holisticai.explainability.metrics.global_feature_importance._surrogate import surrogate_accuracy_score
from holisticai.explainability.metrics.surrogate._classification import (
    AccuracyDegradation,
    surrogate_accuracy_degradation,
    surrogate_fidelity_classification,
)
from holisticai.explainability.metrics.surrogate._clustering import (
    clustering_surrogate_explainability_metrics,
)
from holisticai.explainability.metrics.surrogate._regression import (
    MSEDegradation,
    # regression_surrogate_explainability_metrics,#
    surrogate_fidelity,
    surrogate_fidelity_regression,
    surrogate_mean_squared_error_degradation,
)
from holisticai.explainability.metrics.surrogate._stability import (
    FeatureImportancesStability,
    FeaturesStability,
    surrogate_feature_importances_stability,
    surrogate_features_stability,
)

__all__ = [
    "AccuracyDegradation",
    "surrogate_accuracy_degradation",
    "surrogate_accuracy_score",
    "FeaturesStability",
    "FeatureImportancesStability",
    "surrogate_features_stability",
    "surrogate_feature_importances_stability",
    "classification_surrogate_explainability_metrics",
    "regression_surrogate_explainability_metrics",
    "MSEDegradation",
    "surrogate_fidelity",
    "surrogate_mean_squared_error_degradation",
    "clustering_surrogate_explainability_metrics",
    "surrogate_fidelity_regression",
    "surrogate_fidelity_classification",
]

import pandas as pd


def classification_surrogate_explainability_metrics(Xt_test, yt_test, y_pred, surrogate):
    y_surrogate = surrogate.predict(Xt_test)
    metrics = pd.DataFrame(
        index=["Accuracy Degradation", "Surrogate Fidelity", "Surrogate Feature Stability"],
        columns=["Value", "Reference"],
    )
    metrics.at["Accuracy Degradation", "Value"] = surrogate_accuracy_degradation(yt_test, y_pred, y_surrogate)
    metrics.at["Accuracy Degradation", "Reference"] = 0

    metrics.at["Surrogate Fidelity", "Value"] = surrogate_fidelity_classification(y_pred, y_surrogate)
    metrics.at["Surrogate Fidelity", "Reference"] = 1

    metrics.at["Surrogate Feature Stability", "Value"] = surrogate_features_stability(
        Xt_test, y_pred, surrogate, num_bootstraps=10
    )
    metrics.at["Surrogate Feature Stability", "Reference"] = 1
    return metrics


def regression_surrogate_explainability_metrics(Xt_test, yt_test, yt_pred, surrogate):
    y_surrogate = surrogate.predict(Xt_test)
    metrics = pd.DataFrame(
        index=["MSE Degradation", "Surrogate Fidelity", "Surrogate Feature Stability"], columns=["Value", "Reference"]
    )
    metrics.at["MSE Degradation", "Value"] = surrogate_mean_squared_error_degradation(yt_test, yt_pred, y_surrogate)
    metrics.at["MSE Degradation", "Reference"] = 0

    metrics.at["Surrogate Fidelity", "Value"] = surrogate_fidelity_regression(yt_pred, y_surrogate)
    metrics.at["Surrogate Fidelity", "Reference"] = 1

    metrics.at["Surrogate Feature Stability", "Value"] = surrogate_features_stability(
        Xt_test, yt_pred, surrogate, num_bootstraps=10
    )
    metrics.at["Surrogate Feature Stability", "Reference"] = 1
    return metrics
