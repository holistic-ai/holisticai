import sys
sys.path.append('./')

import numpy as np
from sklearn.linear_model import LogisticRegression
from tutorials.explainability.paper_metrics.utils import load_processed_adult
seed = np.random.seed(42)

X_train, X_test, y_train, y_test = load_processed_adult(seed=seed)
model_type='binary_classification'

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# permutation feature importance
from holisticai.explainability import Explainer
explainer = Explainer(based_on='feature_importance',
                      strategy_type='permutation',
                      model_type='binary_classification',
                      model = model, 
                      x = X_test, 
                      y = y_pred)

explainer.bar_plot(max_display=10)
