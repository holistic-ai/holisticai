from holisticai.utils.models.feature_importance.wrappers.sklearn_base import (
    SKLEARN_BASE_WRAPPERS,
)
from holisticai.utils.models.feature_importance.wrappers.tensorflow_base import (
    TENSORFLOW_BASE_WRAPPERS,
)


def model_wrapper(problem_type, model_class):
    if model_class in [
        "sklearn-1.02",
        "lightgbm-3.3.2",
        "catboost-1.0.4",
        "xgboost-1.5.2",
    ]:
        return SKLEARN_BASE_WRAPPERS[problem_type]
    elif model_class in ["tf-2.9"]:
        return TENSORFLOW_BASE_WRAPPERS[problem_type]
