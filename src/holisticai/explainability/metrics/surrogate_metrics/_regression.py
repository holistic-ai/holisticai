from typing import Any, Literal

import pandas as pd

from holisticai.explainability.metrics.global_importance._surrogate import surrogate_mean_squared_error
from holisticai.explainability.metrics.surrogate_metrics._stability import (
    FeatureImportanceSpread,
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
from holisticai.utils import ModelProxy
from holisticai.utils.models.surrogate import Surrogate


class MSEDifference:
    reference: float = 0
    name: str = "MSE Difference"

    def __call__(self, X, y, proxy, surrogate):
        y_pred = proxy.predict(X)
        Pb = surrogate_mean_squared_error(y, y_pred)
        y_pred_surrogate = surrogate.predict(X)
        Pt = surrogate_mean_squared_error(y, y_pred_surrogate)
        D = Pb - Pt
        return D


def surrogate_mean_squared_error_difference(X, y, proxy, surrogate):
    m = MSEDifference()
    return m(X, y, proxy, surrogate)



def regression_explainability_metrics(proxy:ModelProxy, surrogate: Surrogate, X: Any, y: Any, metric_type:Literal["performance","stability","tree","all"]="all"):

    assert proxy.learning_task == "regression", "Proxy model must be a regression model"

    results = {}
    stratify = y
    is_all = metric_type == "all"
    if is_all or metric_type == 'performance':
        m = MSEDifference()
        results[m.name] = {'Value': m(X, y, proxy, surrogate), 'Reference': m.reference}

        results['Surrogate MSE'] = {'Value': surrogate_mean_squared_error(proxy.predict(X), surrogate.predict(X)), 'Reference': 0}

    if is_all or metric_type == 'stability':
        m = FeaturesStability()
        results[m.name] = {'Value': m(proxy, surrogate, X, stratify=stratify), 'Reference': m.reference}

        m = FeatureImportancesStability()
        results[m.name] = {'Value': m(proxy, surrogate, X, stratify=stratify), 'Reference': m.reference}

        m = FeatureImportanceSpread()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

    if is_all or metric_type == 'tree':
        m = TreeNumberOfFeatures()
        results[m.name] = {'Value': m(surrogate) , 'Reference': m.reference}

        m = TreeNumberOfRules()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

        m = TreeDepthVariance()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

        m = WeightedAverageExplainabilityScore()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

        m = WeightedAverageDepth()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

    return pd.DataFrame(results).T

