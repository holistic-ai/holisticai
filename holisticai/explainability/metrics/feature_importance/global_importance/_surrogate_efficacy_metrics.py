import numpy as np
import pandas as pd
from holisticai.efficacy.metrics import regression_efficacy_metrics, classification_efficacy_metrics

def compute_surrogate_efficacy_metrics(model_type, x, y, surrogate):
    """
    Compute surrogate efficacy metrics for a given model type, model, input features and predicted output.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        x (pandas.DataFrame): The input features.
        surrogate (sklearn estimator): The surrogate model.

    Returns:
        pandas.DataFrame: The surrogate efficacy metrics.
    """

    prediction = surrogate.predict(x)

    if model_type == "binary_classification":
        metric = {
            "Surrogate Efficacy Classification": classification_efficacy_metrics(prediction, y).loc["Accuracy"]
        }

    elif model_type == "regression":
        metric = {
            "Surrogate Efficacy Regression": regression_efficacy_metrics(prediction, y).loc['SMAPE']
        }

    metric = pd.DataFrame(metric)
    return metric.rename(columns={0: "Value"}).T[["Value"]]
