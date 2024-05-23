from holisticai.metrics.efficacy import (
    classification_efficacy_metrics,
    regression_efficacy_metrics,
)


class SurrogacyMetric:
    def __init__(self, model_type):
        if model_type == "binary_classification":
            from sklearn.metrics import accuracy_score

            self.efficacy_metric = accuracy_score
            self.reference = 1
        else:
            from holisticai.metrics.efficacy import smape

            self.efficacy_metric = smape
            self.reference = 0

        self.name = "Surrogacy Efficacy"

    def __call__(self, surrogate, x, y):
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

        return {self.name: self.efficacy_metric(y, prediction)}
