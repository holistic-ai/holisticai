from typing import Optional

import numpy as np
import pandas as pd
import torch

from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

from .dataset import ADDataset
from .models import ADModel, ClassifierModel
from .trainer import TrainArgs, Trainer


class AdversarialDebiasing(BMImp):
    """
    Adversarial debiasing learns a classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the adversary can exploit.

    Obs: Pytorch must be installed in order to use this techinique (pytorch = ">=1.12.1").

    References:
        B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
        Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
        Intelligence, Ethics, and Society, 2018.
    """

    def __init__(
        self,
        features_dim: int = None,
        keep_prob: Optional[float] = 0.1,
        hidden_size: Optional[int] = 128,
        batch_size: Optional[int] = 32,
        shuffle: Optional[bool] = True,
        epochs: Optional[int] = 10,
        initial_lr: Optional[float] = 0.01,
        use_debias: Optional[bool] = True,
        adversary_loss_weight: Optional[float] = 0.1,
        verbose: Optional[int] = 1,
        print_interval: Optional[int] = 100,
        device: Optional[str] = "cpu",
        seed: Optional[int] = None,
    ):

        """
        Parameters
        ----------
        features_dim: int
            Number of input feature X: (n_samples, features_dim)

        keep_prob: float
            Dropout parameter for classifier

        hidden_size: int
            Number of neurons on hidden layer

        batch_size: int
            Numer of examples used for each iteration

        shuffle: bool
            Shuffle data after each epoch

        epochs: int
            Number of epochs

        initial_lr: float
            Initial Learning Rate

        use_debias: bool
            If False Train a simple classifier

        adversary_loss_weight: float
            Adversarial Loss importance

        verbose : int
            Log progress if value > 0.

        print_interval : int
            Each `print_interval` steps print information.

        device: str
            pytorch paramter ("cpu", "cuda")

        seed: int
            seed for random state

        Return
        ------
        self
        """
        # default classifier config
        self.features_dim = features_dim
        self.keep_prob = keep_prob
        self.hidden_size = hidden_size

        # training config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.initial_lr = initial_lr

        # bias config
        self.adversary_loss_weight = adversary_loss_weight
        self.use_debias = use_debias

        # other configs
        self.verbose = verbose
        self.print_interval = print_interval
        self.device = device
        self.seed = seed

        self.sens_groups = SensitiveGroups()

    def transform_estimator(self, estimator=None):
        if estimator is None:
            self.estimator = ClassifierModel(
                self.features_dim, self.hidden_size, self.keep_prob
            )
        else:
            self.estimator = estimator
        return self

    def _create_dataset(self, X, y, sensitive_features):
        """
        Organize data for pytorch environment.
        """
        groups_num = np.array(
            self.sens_groups.fit_transform(sensitive_features, convert_numeric=True)
        )
        dataset = ADDataset(X, y, groups_num, device=self.device)
        return dataset

    def _build_trainer(self, dataset):
        """
        Create a model trainer.
        """

        adm = ADModel(estimator=self.estimator).to(self.device)

        train_args = TrainArgs(
            epochs=self.epochs,
            adversary_loss_weight=self.adversary_loss_weight,
            batch_size=self.batch_size,
            shuffle_data=self.shuffle,
            verbose=self.verbose,
            print_interval=self.print_interval,
        )

        trainer = Trainer(
            model=adm,
            dataset=dataset,
            train_args=train_args,
            use_debias=self.use_debias,
        )
        return trainer

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit the model

        Description
        -----------
        Learn a fair classifier.

        Parameters
        ----------

        X : numpy array
            input matrix

        y_true : numpy array
            Target vector

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector

        Returns
        -------
        the same object
        """
        params = self._load_data(y_true=y_true, group_a=group_a, group_b=group_b)
        y_true = params["y_true"]
        group_a = params["group_a"]
        group_b = params["group_b"]
        self.classes_ = params["classes_"]
        sensitive_features = np.stack([group_a, group_b], axis=1)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        dataset = self._create_dataset(X, y_true, sensitive_features)
        self.trainer = self._build_trainer(dataset)

        self.trainer.train()

        return self

    def predict(self, X):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : np.ndarray
            input matrix

        Returns
        -------

        np.ndarray: Predicted output per sample.
        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1).ravel()

    def predict_proba(self, X):
        """
        Prediction

        Description
        ----------
        Predict matrix probability for the given samples.

        Parameters
        ----------
        X : np.ndarray
            input matrix

        Returns
        -------

        np.ndarray: Predicted matrix probability per sample.
        """
        proba = np.empty((X.shape[0], 2))
        proba[:, 1] = self._forward(X)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    def predict_score(self, X):
        """
        Prediction

        Description
        ----------
        Predict probability for the given samples.

        Parameters
        ----------
        X : np.ndarray
            input matrix

        Returns
        -------

        np.ndarray: Predicted probability per sample.
        """
        p = self._forward(X).reshape([-1])
        return p

    def _preprocessing_data(self, X):
        X = np.array(X)
        X = torch.tensor(X).type(torch.FloatTensor)[None, :]
        return X

    def _forward(self, X):
        X = self._preprocessing_data(X)
        with torch.no_grad():
            p = self.estimator(X, trainable=False).ravel()
        return p
