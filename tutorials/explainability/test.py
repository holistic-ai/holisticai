import sys
sys.path.insert(0, '/home/cristian/github/holisticai')

import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from holisticai.explainability import Explainer
from holisticai.efficacy.metrics import classification_efficacy_metrics
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from holisticai.explainability.metrics.global_importance._contrast_metrics import important_constrast_matrix
from utils import load_processed_adult , train_classifier_model, train_regression_model, load_processed_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from utils import run_permutation_explainability, run_surrogate_explainability, run_shap_explainability, run_lime_explainability
seed = np.random.seed(42)

X_train, X_test, y_train, y_test = load_processed_diabetes(seed=seed)
model_type='regression'
outputs = train_regression_model(X_train, X_test, y_train, y_test)

print("Permutation")
run_permutation_explainability(model_type, outputs)
print("Surrogate")
run_surrogate_explainability(model_type, outputs)
print("Shap")
run_shap_explainability(model_type, outputs)
print("Lime")
run_lime_explainability(model_type, outputs)


X_train, X_test, y_train, y_test = load_processed_adult(seed=seed)
model_type='binary_classification'
outputs = train_classifier_model(X_train, X_test, y_train, y_test)
print("Permutation")
run_permutation_explainability(model_type, outputs)
print("Surrogate")
run_surrogate_explainability(model_type, outputs)
print("Shap")
run_shap_explainability(model_type, outputs)
print("Lime")
run_lime_explainability(model_type, outputs)


