from holisticai.utils.models.metrics import BinaryClassificationMetrics, SimpleRegressionMetrics
from holisticai.utils.models.feature_importance.wrappers import model_wrapper

import pandas as pd
import numpy as np

METRICS = {
    'binary_classification': BinaryClassificationMetrics(),
    'regression': SimpleRegressionMetrics(),
}

def inference(model_type, wdt, x):
    """
    Compute inference for a given model type, surrogate model and input features.

    Args:
        model_type (str): The type of the model, either 'binary_classification' or 'regression'.
        wdt (holisticai.utils.models.feature_importance.wrappers.ModelWrapper): The surrogate model.
        x (pandas.DataFrame): The input features.

    Returns:
        dict: The inference results.
    """
    if model_type == 'binary_classification':
        return {"yproba": wdt.predict_proba(x), "ypred": wdt.predict(x)}
    elif model_type == 'regression':
        return {"ypred": wdt.predict(x)}
    else:
        raise ValueError("model_type must be either 'binary_classification' or 'regression'")

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

    eff_metric = METRICS[model_type]
    wp = model_wrapper(problem_type = model_type, model_class='sklearn-1.02')
    wdt = wp(surrogate)
    wdt.mode('efficacy')
    prediction = inference(model_type, wdt, x)
    if wdt._estimator_type == 'classifier':
        num_classes = len(wdt.classes_)
        if num_classes == 2:
            class_dict = dict(zip(wdt.classes_, range(num_classes)))
            classes_id_vect = np.vectorize(lambda lbl : class_dict[lbl])
            y = classes_id_vect(y).ravel()
            prediction['ypred'] = classes_id_vect(prediction['ypred']).ravel()
    
    if model_type == 'binary_classification':
        metric = {'Surrogate Efficacy Classification': eff_metric(y, **prediction).T.reset_index(drop=True)['Accuracy']}
    
    elif model_type == 'regression':
        metric = {'Surrogate Efficacy Regression': eff_metric(y, **prediction).T.reset_index(drop=True)['RMSE']}
    
    metric = pd.DataFrame(metric)
    return metric.rename(columns={0:'Value'})
