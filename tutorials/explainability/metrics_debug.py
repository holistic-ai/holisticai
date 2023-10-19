import sys

sys.path.insert(0, "./")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_adult, load_us_crime

# Dataset
dataset = load_adult()

# Dataframe
df = pd.concat([dataset["data"], dataset["target"]], axis=1)
# df = df.iloc[:30, :]
protected_variables = ["sex", "race"]
output_variable = ["class"]

# Simple preprocessing
y = df[output_variable].replace({">50K": 1, "<=50K": 0})
X = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1), dtype=float)

group = ["sex"]
group_a = df[group] == "Female"
group_b = df[group] == "Male"
data = [X, y, group_a, group_b]

# Train test split
dataset = train_test_split(*data, test_size=0.2, shuffle=True)
train_data = dataset[::2]
test_data = dataset[1::2]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # train test split

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

seed = np.random.seed(42)  # set seed for reproducibility

model = GradientBoostingClassifier()  # instantiate model
# model = LinearRegression() # instantiate model
model.fit(X_train, y_train)  # fit model

y_pred = model.predict(X_test)  # compute predictions

# import Explainer
from holisticai.explainability import Explainer

explainer = Explainer(
    based_on="feature_importance",
    strategy_type="permutation",
    model_type="binary_classification",
    model=model,
    x=X_test,
    y=y_pred,
)

explainer.metrics(alpha=0.7)
