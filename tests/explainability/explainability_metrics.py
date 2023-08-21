import sys
sys.path.append("./")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_adult

# data and simple preprocessing
dataset = load_adult()['frame']
X = pd.get_dummies(dataset.iloc[:1000,:].drop(columns=['class']), drop_first=True)
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

y_clf = pd.DataFrame(dataset.iloc[:1000,:]['class'].apply(lambda x: 1 if x == '>50K' else 0))
y_reg = pd.DataFrame(dataset.iloc[:1000,:]['fnlwgt'])
y_reg = scaler.fit_transform(y_reg)

# instantiate and fit models
reg = LinearRegression()
reg.fit(X_standard, y_reg)

clf = LogisticRegression(random_state=42, max_iter=100)
clf.fit(X_standard, y_clf)

from holisticai.explainability import Explainer

# instantiate explainer lime classification
explainer = Explainer(based_on='feature_importance', 
                      strategy_type='lime', 
                      model_type='binary_classification', 
                      model=clf, 
                      x=X_standard, 
                      y=y_clf)

explainer.metrics()
