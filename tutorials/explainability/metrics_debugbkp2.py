import sys

sys.path.append("./")

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "../../")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_adult, load_us_crime
from holisticai.efficacy.metrics import regression_efficacy_metrics

# import Explainer
from holisticai.explainability import Explainer

dataset = load_us_crime(return_X_y=False, as_frame=True)
df = pd.concat([dataset["data"], dataset["target"]], axis=1)


def preprocess_us_crime_dataset(df, protected_feature):
    """Performs the pre-processing step of the data."""
    # Remove NaN elements from dataframe
    df_ = df.copy()
    df_clean = df_.iloc[
        :, [i for i, n in enumerate(df_.isna().sum(axis=0).T.values) if n < 1000]
    ]
    df_clean = df_clean.dropna()
    # Get the protected attribute vectors
    group_a = df_clean[protected_feature].apply(lambda x: x > 0.5)
    group_b = 1 - group_a
    group_b = group_b.astype("bool")
    # Remove unnecessary columns
    cols = [
        c
        for c in df_clean.columns
        if (not c.startswith("race")) and (not c.startswith("age"))
    ]
    df_clean = df_clean[cols].iloc[:, 3:]
    return df_clean, group_a, group_b


df_clean, group_a, group_b = preprocess_us_crime_dataset(df, "racePctWhite")
X = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # train test split

model = GradientBoostingRegressor()  # instantiate model
# model = LinearRegression() # instantiate model
model.fit(X_train, y_train)  # fit model

y_pred = model.predict(X_test)  # compute predictions

# instantiate explainer permutation
# permutation feature importance
explainer = Explainer(
    based_on="feature_importance",
    strategy_type="permutation",
    model_type="regression",
    model=model,
    x=X,
    y=y,
)

print(explainer.metrics())
