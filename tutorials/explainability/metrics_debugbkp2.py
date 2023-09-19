import sys

sys.path.append("./")

import warnings

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '../../')

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from holisticai.datasets import load_us_crime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from holisticai.explainability import Explainer
from holisticai.efficacy.metrics import regression_efficacy_metrics

from holisticai.datasets import load_adult
# import Explainer
from holisticai.explainability import Explainer

dataset = load_diabetes()  # load dataset

X = dataset.data  # features
y = dataset.target  # target
feature_names = dataset.feature_names  # feature names

X = pd.DataFrame(X, columns=feature_names)  # convert to dataframe
# data and simple preprocessing
dataset = load_adult()["frame"]
# dataset = dataset.iloc[0:1000,]

seed = np.random.seed(42)  # set seed for reproducibility
# simple preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)  # train test split

model = GradientBoostingRegressor()  # instantiate model
# model = LinearRegression() # instantiate model
model.fit(X_train, y_train)  # fit model

y_pred = model.predict(X_test)  # compute predictions

# instantiate explainer permutation
explainer = Explainer(
    based_on="feature_importance",
    strategy_type="lime",
    model_type="regression",
    model=model,
    x=X,
    y=y,
)

print(explainer.metrics(detailed=True))
