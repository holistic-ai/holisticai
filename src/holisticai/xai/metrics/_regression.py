from typing import Literal

from holisticai.datasets import Dataset
from holisticai.xai.commons import (
    RegressionClassificationXAISettings,
    compute_xai_features,
    select_feature_importance_strategy,
)
from holisticai.xai.metrics._utils import compute_xai_metrics_from_features


def regression_xai_features(X, y, predict_fn, strategy : Literal["permutation","surrogate"]="permutation"):
    dataset = Dataset(X=X, y=y)
    learning_task_settings = RegressionClassificationXAISettings(predict_fn=predict_fn)
    feature_importance_fn = select_feature_importance_strategy(learning_task_settings=learning_task_settings, strategy=strategy)
    return compute_xai_features(dataset=dataset, learning_task_settings=learning_task_settings, feature_importance_fn=feature_importance_fn)

def regression_xai_metrics(X, y, predict_fn, strategy: Literal["permutation","surrogate"]="permutation"):
    xai_features = regression_xai_features(X, y, predict_fn, strategy=strategy)
    return compute_xai_metrics_from_features(xai_features)
