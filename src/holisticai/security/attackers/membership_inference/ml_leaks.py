import numpy as np
from sklearn.base import BaseEstimator, clone


class MLleaks:
    """
    Class for performing Membership Inference Attacks (MIA) using shadow models.

    This class is designed to train a shadow model on a shadow dataset and then \
    use the shadow model to create a membership inference dataset. The class \
    supports only binary classification task.

    The procedure follows the implementation of the MIA attack as described in \
    "ML-Leaks: Model and Data Independent Membership Inference Attacks and \
    Defenses on Machine Learning Models" by Salem et al. (2018).

    Parameters
    ----------
    target_model : BaseEstimator
        The target model to be attacked. It should be an instance of a scikit-learn \
        estimator that implements the `predict_proba` method.
    target_dataset : tuple
        A tuple containing the target training and testing datasets. Each dataset \
        should be a tuple of (X, y), where X is the feature matrix and y is the target vector.
    shadow_dataset : tuple
        A tuple containing the shadow training and testing datasets. Each dataset \
        should be a tuple of (X, y), where X is the feature matrix and y is the target vector.
    task : str, optional
        The type of task to perform. It can be either "binary" or "multiclass". \
        Default is "binary".

    References
    ----------

    .. [1] Salem, A. et al. (2018). ML-Leaks: Model and Data Independent \
    Membership Inference Attacks and Defenses on Machine Learning Models.
    """

    def __init__(self, target_model: BaseEstimator, target_dataset: tuple, shadow_dataset: tuple, task: str = "binary"):
        if not isinstance(target_model, BaseEstimator):
            raise TypeError("target_model must be an instance of BaseEstimator.")

        if not isinstance(target_dataset, tuple) or len(target_dataset) != 2:
            raise ValueError("target_dataset must be a tuple with two elements (train and test datasets).")

        if not isinstance(shadow_dataset, tuple) or len(shadow_dataset) != 2:
            raise ValueError("shadow_dataset must be a tuple with two elements (train and test datasets).")

        if task not in ["binary", "multiclass"]:
            raise ValueError("task must be either 'binary' or 'multiclass'.")

        self.task = task
        self.target_model = target_model
        self.target_dataset = target_dataset
        self.shadow_dataset = shadow_dataset

    def fit(self) -> tuple:
        """
        Trains the shadow model and generates the membership inference dataset to train the attacker model.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - (X_mia_train, y_mia_train): Training dataset for the attacker model.
            - (X_mia_test, y_mia_test): Testing dataset for the attacker model.
        """
        target_train, target_test = self.target_dataset
        X_target_train, _ = target_train
        X_target_test, _ = target_test
        X_mia_train, y_mia_train = self._train_shadow_model()
        target_train_preds, target_test_preds = self._get_probs(self.target_model, X_target_train, X_target_test)
        X_mia_test, y_mia_test = self._create_attacker_dataset(target_train_preds, target_test_preds)
        return (X_mia_train, y_mia_train), (X_mia_test, y_mia_test)

    def _get_probs(self, model: BaseEstimator, X_train: np.ndarray, X_test: np.ndarray) -> tuple:
        """
        Computes the predicted probabilities for the training and testing datasets
        using the provided model.

        Parameters
        ----------
        model : BaseEstimator
            A trained model that implements the `predict_proba` method.
        X_train : array-like of shape (n_samples_train, n_features)
            The training data for which probabilities are to be predicted.
        X_test : array-like of shape (n_samples_test, n_features)
            The testing data for which probabilities are to be predicted.

        Returns
        -------
        tuple of ndarray
            A tuple containing:
            - y_train_pred : ndarray of shape (n_samples_train, n_classes)
              Predicted probabilities for the training data.
            - y_test_pred : ndarray of shape (n_samples_test, n_classes)
              Predicted probabilities for the testing data.
        """
        y_train_pred = model.predict_proba(X_train)
        y_test_pred = model.predict_proba(X_test)
        return y_train_pred, y_test_pred

    def train_model(self, X_data: np.ndarray, y_data: np.ndarray) -> BaseEstimator:
        """
        Trains a clone of the target model using the provided data.

        Parameters
        ----------
        X_data : np.ndarray
            The input features for training the model.
        y_data : np.ndarray
            The target labels corresponding to the input features.

        Returns
        -------
        BaseEstimator
            A trained instance of the target model.
        """
        model = clone(self.target_model)
        model.fit(X_data, y_data)
        return model

    def _train_shadow_model(self) -> tuple:
        """
        Trains the shadow model and creates the attacker dataset.

        This method trains a shadow model using the shadow dataset, obtains \
        predictions for both the shadow training and testing data, and then \
        creates a dataset for the attacker model based on these predictions.

        Returns
        -------
        tuple
            A tuple containing the attacker dataset created from the shadow \
            model's predictions on the shadow training and testing data.
        """
        shadow_train, shadow_test = self.shadow_dataset
        X_shadow_train, y_shadow_train = shadow_train
        X_shadow_test, _ = shadow_test
        self.shadow_model = self.train_model(X_shadow_train, y_shadow_train)
        shadow_train_preds, shadow_test_preds = self._get_probs(self.shadow_model, X_shadow_train, X_shadow_test)
        return self._create_attacker_dataset(shadow_train_preds, shadow_test_preds)

    def _create_attacker_dataset(self, train_preds: np.ndarray, test_preds: np.ndarray, shuffle: bool = True) -> tuple:
        """
        Creates a dataset for training a membership inference attacker model.

        This function combines predictions from the training and test datasets\
        to create features (`X_mia`) and labels (`y_mia`) for the attacker model.\
        Labels are assigned as 1 for training predictions and 0 for test predictions.\
        Optionally, the resulting dataset can be shuffled.

        Parameters
        ----------
        train_preds : np.ndarray
            Predictions from the training dataset. Shape should match the model's output.
        test_preds : np.ndarray
            Predictions from the test dataset. Shape should match the model's output.
        shuffle : bool, optional
            Whether to shuffle the resulting dataset, by default True.

        Returns
        -------
        tuple
            A tuple containing:
            - X_mia (np.ndarray): Combined predictions from training and test datasets.
            - y_mia (np.ndarray): Labels indicating membership (1 for training, 0 for test).

        Raises
        ------
        NotImplementedError
            If the task is "multiclass", as this functionality is not implemented yet.
        """
        if self.task == "binary":
            X_mia = np.concatenate((train_preds, test_preds), axis=0)
            y_mia = np.concatenate((np.ones(len(train_preds)), np.zeros(len(test_preds))), axis=0)
        elif self.task == "multiclass":
            raise NotImplementedError("Multiclass MIA is not implemented yet.")
        if shuffle:
            indices = np.arange(len(X_mia))
            np.random.shuffle(indices)
            X_mia = X_mia[indices]
            y_mia = y_mia[indices]
        return X_mia, y_mia
