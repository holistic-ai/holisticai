from holisticai.utils.models.feature_importance.wrappers.sklearn_base import binary_classification, simple_regression
from holisticai.utils.models.parameters import PROBLEM_TYPES

SKLEARN_BASE_WRAPPERS = {
    PROBLEM_TYPES.BINARY_CLASSIFICATION: binary_classification.WBinaryClassificationModel,
    PROBLEM_TYPES.SIMPLE_REGRESSION: simple_regression.WSimpleRegressionModel,
}
