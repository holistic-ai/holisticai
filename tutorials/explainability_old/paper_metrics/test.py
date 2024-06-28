import sys

sys.path.append("../../")

import numpy as np

from tutorials.explainability.paper_metrics.utils import (
    load_processed_adult,
    load_processed_parkinson,
    run_lime_explainability,
    run_permutation_explainability,
    run_shap_explainability,
    run_surrogate_explainability,
    train_classifier_model,
    train_regression_model,
)

seed = np.random.seed(42)

X_train, X_test, y_train, y_test = load_processed_parkinson(seed=seed)
model_type = "regression"
outputs = train_regression_model(X_train, X_test, y_train, y_test)

print("Permutation")
run_permutation_explainability(model_type, outputs)

"""
print("Surrogate")
run_surrogate_explainability(model_type, outputs)
print("Shap")
run_shap_explainability(model_type, outputs)
print("Lime")
run_lime_explainability(model_type, outputs)

X_train, X_test, y_train, y_test = load_processed_adult(seed=seed)
model_type = "binary_classification"
outputs = train_classifier_model(X_train, X_test, y_train, y_test)
print("Permutation")
run_permutation_explainability(model_type, outputs)
print("Surrogate")
run_surrogate_explainability(model_type, outputs)
print("Shap")
run_shap_explainability(model_type, outputs)
print("Lime")
run_lime_explainability(model_type, outputs)
"""
