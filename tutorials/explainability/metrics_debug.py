import sys

sys.path.append("./")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_adult

# data and simple preprocessing
dataset = load_adult()["frame"]
dataset = dataset.iloc[
    0:1000,
]

X = pd.get_dummies(dataset.drop(columns=["class", "fnlwgt"]), drop_first=True)
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
X_standard = pd.DataFrame(X_standard, columns=X.columns)

y_clf = pd.DataFrame(dataset["class"].apply(lambda x: 1 if x == ">50K" else 0))
y_reg = pd.DataFrame(dataset["fnlwgt"])
y_reg = scaler.fit_transform(y_reg)

# regression
reg = LinearRegression()
reg.fit(X_standard, y_reg)

# classification
clf = LogisticRegression(random_state=42, max_iter=100)
clf.fit(X_standard, y_clf)

# import Explainer
from holisticai.explainability import Explainer

# instantiate explainer permutation
explainer = Explainer(
    based_on="feature_importance",
    strategy_type="permutation",
    model_type="binary_classification",
    model=clf,
    x=X_standard,
    y=y_clf,
)

print(explainer.metrics())
