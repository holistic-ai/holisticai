from typing import Any, Literal

import pandas as pd
from sklearn.metrics import accuracy_score

from holisticai.explainability.metrics.global_importance._surrogate import surrogate_accuracy_score
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


class AccuracyDifference:
    reference: float = 0
    name: str = "Accuracy Difference"

    def __call__(self, X, y, proxy, surrogate):
        y_pred = proxy.predict(X)
        Pb = accuracy_score(y, y_pred)
        y_pred_surrogate = surrogate.predict(X)
        Pt = accuracy_score(y, y_pred_surrogate)
        D = Pb - Pt
        return D

def accuracy_difference(X, y, proxy, surrogate):
    m = AccuracyDifference()
    return m(X, y, proxy, surrogate)


def classification_explainability_metrics(proxy: ModelProxy, surrogate:Surrogate, X: Any, y: Any, metric_type:Literal["performance","stability","tree","all"]="all"):

    assert proxy.learning_task in ["binary_classification", "multi_classification"], "Proxy model must be a classification model"

    results = {}
    stratify = y
    is_all = metric_type == "all"
    if is_all or metric_type == 'performance':
        m = AccuracyDifference()
        results[m.name] = {'Value': m(X, y, proxy, surrogate), 'Reference': m.reference}

        results['Surrogate Accuracy'] = {'Value': surrogate_accuracy_score(proxy.predict(X), surrogate.predict(X)), 'Reference': 1}

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

        m = WeightedTreeGini()
        results[m.name] = {'Value': m(surrogate), 'Reference': m.reference}

    return pd.DataFrame(results).T
