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

import pandas as pd
import shap
import sklearn

# a classic housing price dataset
X,y = shap.datasets.california(n_points=1000)

X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
# compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X).values

y_pred = model.predict(X)

from holisticai.explainability.metrics.utils import LimeTabularHandler, ShapTabularHandler
from holisticai.explainability import Explainer

local_explainer_handler = LimeTabularHandler(
        model.predict,
        X.values,
        feature_names=X.columns.tolist(),
        discretize_continuous=True,
        mode='regression')

lime_explainer_metrics = Explainer(based_on='feature_importance',
                      strategy_type='local',
                      model_type='regression',
                      x = X, 
                      y = y_pred,
                      local_explainer_handler=local_explainer_handler)

lime_explainer_metrics.metrics(detailed=True)