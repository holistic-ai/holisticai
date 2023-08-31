from holisticai.utils.models.feature_importance.wrappers.tensorflow_base import (
    binary_classification,
)
from holisticai.utils.models.parameters import PROBLEM_TYPES

TENSORFLOW_BASE_WRAPPERS = {
    PROBLEM_TYPES.BINARY_CLASSIFICATION: binary_classification.TFBinaryClassificationModel,
}
