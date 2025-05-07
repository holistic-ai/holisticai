from typing import Any, Literal

import numpy as np
import pandas as pd

from holisticai.explainability.metrics.global_feature_importance._importance_spread import (
    FeatureImportanceSpread,
)
from holisticai.explainability.metrics.global_feature_importance._surrogate import (
    surrogate_fidelity,
    surrogate_mean_squared_error,
)
from holisticai.explainability.metrics.surrogate._stability import (
    FeatureImportancesStability,
    FeaturesStability,
)
from holisticai.explainability.metrics.tree._tree import (
    TreeDepthVariance,
    TreeNumberOfFeatures,
    TreeNumberOfRules,
    WeightedAverageDepth,
    WeightedAverageExplainabilityScore,
)
from holisticai.typing import ArrayLike
from holisticai.utils.surrogate_models import RegressionSurrogate


class MSEDegradation:
    reference: float = 0
    name: str = "MSE Degradation"

    def __call__(self, y, y_pred, y_surrogate):
        Pb = surrogate_mean_squared_error(y, y_pred)
        Ps = surrogate_mean_squared_error(y, y_surrogate)
        return max(0, 2 * (Ps - Pb) / (Pb + Ps))


def surrogate_mean_squared_error_degradation(y: ArrayLike, y_pred: ArrayLike, y_surrogate: ArrayLike):
    """
    Calculate the difference between the mean squared error of the original model and the surrogate model.

    Parameters
    ----------

    y : ArrayLike
        The true target values.

    y_pred : ArrayLike
        The predicted target values of the original model.

    y_surrogate : ArrayLike
        The predicted target values of the surrogate model.

    Returns
    -------
    float
        The difference between the mean squared error of the original model and the surrogate model

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.explainability.metrics.surrogate import (
    ...     surrogate_smape_difference,
    ... )
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    >>> y_surrogate = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
    >>> surrogate_smape_difference(y, y_pred, y_surrogate)
    """
    m = MSEDegradation()
    return m(y, y_pred, y_surrogate)


class SurrogateFidelityRegression:
    """
    FeaturesStability calculates the stability of features used in a surrogate model.
    The metric measures the similarity of features used in the surrogate model across different bootstraps.

    Parameters
    ----------
        reference (float): The reference of best stability value = 1.
        name (str): The name of the stability metric: "Features Stability".
    """

    reference: float = 1
    name: str = "Surrogate Fidelity Regression"

    def __call__(self, y_pred, y_surrogate):
        # return surrogate_fidelity(y_pred, y_surrogate)
        epsilon = 1e-10
        # Normalizar el error absoluto entre y_pred y y_surrogate
        abs_error = np.abs(y_pred - y_surrogate)
        max_value = np.maximum(np.abs(y_pred), np.abs(y_surrogate)) + epsilon

        # Calcular el error relativo normalizado
        relative_error = abs_error / max_value

        # Devolver 1 menos el error promedio, lo que representa la fidelidad
        return 1 - np.mean(relative_error)


def surrogate_fidelity_regression(y_pred, y_surrogate):
    """
    Calculate the surrogate fidelity for regression models.

    This function evaluates how well a surrogate model's predictions match the
    predictions of the original model.

    Parameters:
    y_pred (array-like): Predictions from the original model.
    y_surrogate (array-like): Predictions from the surrogate model.

    Returns:
    float: A fidelity score indicating how closely the surrogate model's
           predictions match the original model's predictions.
    """
    m = SurrogateFidelityRegression()
    return m(y_pred, y_surrogate)


def regression_surrogate_explainability_metrics(
    X: Any,
    y: Any,
    y_pred: Any,
    surrogate_type: Literal["shallow_tree", "tree"],
    metric_type: Literal["performance", "stability", "tree", "all"] = "all",
    return_surrogate_model: bool = False,
):
    surrogate = RegressionSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    y_surrogate = surrogate.predict(X)

    results = {}
    is_all = metric_type == "all"
    if is_all or metric_type == "performance":
        m = MSEDegradation()
        results[m.name] = {"Value": m(y, y_pred, y_surrogate), "Reference": m.reference}

        results["Surrogate Fidelity"] = {"Value": surrogate_fidelity(y_pred, y_surrogate), "Reference": 0}

    if is_all or metric_type == "stability":
        m = FeaturesStability()
        results[m.name] = {"Value": m(X, y_pred, surrogate), "Reference": m.reference}

        m = FeatureImportancesStability()
        results[m.name] = {"Value": m(X, y_pred, surrogate), "Reference": m.reference}

        m = FeatureImportanceSpread()
        results[m.name] = {"Value": m(surrogate.feature_importances_), "Reference": m.reference}

    if is_all or metric_type == "tree":
        m = TreeNumberOfFeatures()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = TreeNumberOfRules()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = TreeDepthVariance()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = WeightedAverageExplainabilityScore()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = WeightedAverageDepth()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

    if return_surrogate_model:
        return pd.DataFrame(results).T, surrogate
    return pd.DataFrame(results).T
