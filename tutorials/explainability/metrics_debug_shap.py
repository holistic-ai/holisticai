import sys

sys.path.insert(0, "./")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_adult, load_us_crime

# Dataset
dataset = load_adult()

# Dataframe
df = pd.concat([dataset["data"], dataset["target"]], axis=1)
protected_variables = ["sex", "race"]
output_variable = ["class"]

# Simple preprocessing
y = df[output_variable].replace({">50K": 1, "<=50K": 0})
X = pd.get_dummies(
    df.drop(protected_variables + output_variable, axis=1), dtype="float"
)

y = y.iloc[:100, :]
X = X.iloc[:100, :]

# a simple linear model
model = LogisticRegression()
model.fit(X, y)

from holisticai.explainability import Explainer

explainer = Explainer(
    based_on="feature_importance",
    strategy_type="lime",
    model_type="binary_classification",
    model=model,
    x=X,
    y=y,
)

import matplotlib.pyplot as plt

explainer.show_features_stability_boundaries(ncols=2, figsize=(15, 5))
plt.show()
explainer.show_data_stability_boundaries(top_n=10, ncols=2, figsize=(15, 5))
plt.show()
