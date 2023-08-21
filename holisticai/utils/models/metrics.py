from holisticai.efficacy.metrics import rmse_score
from sklearn import metrics

import pandas as pd

class BaseMetrics:
    def __call__(self, yobs, ypred, yproba=None):
        ### Supervised Learning (Regression, Classification, etc.)
        # yobs - target / dependent variable
        # ypred - model predictions (hard predictions)
        # yproba - model scorings (soft predictions, like probabilities)
        metrics = [[pf, fn(yobs, ypred, **self.hyper[pf])] for pf, fn in self.perform.items()]
        if yproba is not None:
            metrics += [[pf, fn(yobs, yproba, **self.soft_hyper[pf])] for pf, fn in self.soft_perform.items()]

        return pd.DataFrame(metrics, columns=["Metric", "Value"]).set_index('Metric')

class BinaryClassificationMetrics(BaseMetrics):
    def __init__(self, **kargs):
        self.perform = \
            {
                'Accuracy': metrics.accuracy_score,
                "Balanced accuracy": metrics.balanced_accuracy_score,
                "Precision": metrics.precision_score,
                "Recall": metrics.recall_score,
                "F1-Score": metrics.f1_score
            }

        self.hyper = \
            {
                'Accuracy': {},
                "Balanced accuracy": {},
                "Precision": {},
                "Recall": {},
                "F1-Score": {}
            }

        self.soft_perform = \
            {
                "AUC": metrics.roc_auc_score,
                "Log Loss": metrics.log_loss
            }

        self.soft_hyper = \
            {
                "AUC": {"average": "micro"},
                "Log Loss": {}
            }

class SimpleRegressionMetrics(BaseMetrics):
    def __init__(self, **kargs):
        self.perform = {
            "RMSE": rmse_score,
            "MAE": metrics.mean_absolute_error,
            "MAPE": metrics.mean_absolute_percentage_error,
            "Max Error": metrics.max_error,
        }

        self.hyper = {
            "RMSE": {"sample_weight": None,
                     "multioutput": 'uniform_average'},
            "MAE": {"sample_weight": None,
                    "multioutput": 'uniform_average'},
            "MAPE": {"sample_weight": None,
                     "multioutput": 'uniform_average'},
            "Max Error": {},
        }