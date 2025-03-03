from typing import Any, Literal

import numpy as np
import pandas as pd
from holisticai.explainability.metrics.global_feature_importance._importance_spread import FeatureImportanceSpread
from holisticai.explainability.metrics.global_feature_importance._surrogate import (
    surrogate_accuracy_score,
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
    WeightedTreeGini,
)
from holisticai.typing import ArrayLike
from holisticai.utils.surrogate_models import BinaryClassificationSurrogate, MultiClassificationSurrogate
from sklearn.metrics import accuracy_score


class AccuracyDegradation:
    reference: float = 0
    name: str = "Accuracy Degradation"

    def __call__(self, y, y_pred, y_surrogate):
        Pb = accuracy_score(y, y_pred)
        Ps = accuracy_score(y, y_surrogate)
        return 2 * (Pb - Ps) / (Pb + Ps)  # Normalized difference between the two SMAPE values


def surrogate_accuracy_degradation(y: ArrayLike, y_pred: ArrayLike, y_surrogate: ArrayLike):
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
    m = AccuracyDegradation()
    return m(y, y_pred, y_surrogate)


class SurrogateFidelityClassification:
    """
    FeaturesStability calculates the stability of features used in a surrogate model.
    The metric measures the similarity of features used in the surrogate model across different bootstraps.

    Parameters
    ----------
        reference (float): The reference of best stability value = 1.
        name (str): The name of the stability metric: "Features Stability".
    """

    reference: float = 1
    name: str = "Surrogate Fidelity Classification"

    def __call__(self, y_pred, y_surrogate):
        return accuracy_score(y_pred, y_surrogate)


def surrogate_fidelity_classification(y_pred, y_surrogate):
    """
    Calculate the surrogate fidelity for classification tasks.

    Surrogate fidelity measures how well the surrogate model's predictions
    match the original model's predictions.

    Parameters
    ----------
    y_pred : array-like
        Predictions from the original model.
    y_surrogate : array-like
        Predictions from the surrogate model.

    Returns
    -------
    float
        The surrogate fidelity score.
    """
    m = SurrogateFidelityClassification()
    return m(y_pred, y_surrogate)


def classification_surrogate_explainability_metrics(
    X: Any,
    y: Any,
    y_pred: Any,
    surrogate_type: Literal["shallow_tree", "tree"],
    metric_type: Literal["performance", "stability", "tree", "all"] = "all",
    return_surrogate_model: bool = False,
):
    if len(np.unique(y_pred)) == 2:
        surrogate = BinaryClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    elif len(np.unique(y_pred)) > 2:
        surrogate = MultiClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    else:
        msg = "y_pred must have at least two unique values"
        raise ValueError(msg)

    y_surrogate = surrogate.predict(X)
    results = {}
    is_all = metric_type == "all"
    if is_all or metric_type == "performance":
        m = AccuracyDegradation()
        results[m.name] = {"Value": m(y, y_pred, y_surrogate), "Reference": m.reference}

        results["Surrogate Accuracy"] = {"Value": surrogate_accuracy_score(y_pred, y_surrogate), "Reference": 1}

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

        m = WeightedTreeGini()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

    if return_surrogate_model:
        return pd.DataFrame(results).T, surrogate
    return pd.DataFrame(results).T
