from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from holisticai.bias.mitigation.inprocessing.adversarial_debiasing.models import (
    ADModel,
    AdversarialModel,
    ClassifierModel,
    create_train_state,
    train_step,
)
from holisticai.datasets import DataLoader, Dataset
from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

logger = logging.getLogger(__name__)


def is_numeric(df):
    if isinstance(df, pd.DataFrame):
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
    if isinstance(df, np.ndarray):
        return np.issubdtype(df.dtype, np.number)
    raise ValueError("Input must be a pandas DataFrame or numpy array.")


class AdversarialDebiasing(BMImp):
    """Adversarial Debiasing

    Adversarial debiasing [1]_ learns a classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the adversary can exploit.

    Obs: Pytorch must be installed in order to use this techinique (pytorch = ">=1.12.1").

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

    Methods
    -------
        fit(X, y_true, group_a, group_b)
            Fit model using Adversarial Debiasing.

        predict(X)
            Predict the closest cluster each sample in X belongs to.

        predict_proba(X)
            Predict the probability of each sample in X belongs to each class.

        predict_score(X)
            Predict the probability of each sample in X belongs to the positive class.

    References
    ----------
        [1] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
        Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
        Intelligence, Ethics, and Society, 2018.
    """

    def __init__(
        self,
        features_dim: Optional[int] = None,
        keep_prob: Optional[float] = 0.1,
        hidden_size: Optional[int] = 128,
        batch_size: Optional[int] = 32,
        shuffle: Optional[bool] = True,
        epochs: Optional[int] = 10,
        learning_rate: Optional[float] = 0.01,
        use_debias: Optional[bool] = True,
        adversary_loss_weight: Optional[float] = 0.1,
        verbose: Optional[int] = 1,
        print_interval: Optional[int] = 100,
        device: Optional[str] = "cpu",
        seed: Optional[int] = None,
    ):
        # default classifier config
        self.features_dim = features_dim
        self.keep_prob = keep_prob
        self.hidden_size = hidden_size

        # training config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.learning_rate = learning_rate

        # bias config
        self.adversary_loss_weight = adversary_loss_weight
        self.use_debias = use_debias

        # other configs
        self.verbose = verbose
        self.print_interval = print_interval
        self.device = device
        self.seed = seed if seed is not None else np.random.randint(0, 1000)

        self._sensgroups = SensitiveGroups()

    def transform_estimator(self, estimator=None):
        if estimator is None:
            self.classifier = ClassifierModel(
                features_dim=self.features_dim, hidden_size=self.hidden_size, keep_prob=self.keep_prob
            )
        else:
            self.estimator = estimator
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
        import pandas as pd

        params = self._load_data(X=X, y=y, group_a=group_a, group_b=group_b)
        x = pd.DataFrame(params["X"])
        if not is_numeric(x):
            raise ValueError("Adversarial Debiasing only works with numeric features.")

        y = pd.Series(params["y"])
        group_a = pd.Series(params["group_a"])
        group_b = pd.Series(params["group_b"])
        self.classes_ = params["classes_"]

        dataset = Dataset(X=x, y=y, group=group_a)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, dtype="jax")

        feature_dim = x.shape[1]

        rng = jax.random.PRNGKey(self.seed)
        adversary_model = AdversarialModel()
        model = ADModel(classifier=self.classifier, adversarial=adversary_model)
        cls_state, adv_state = create_train_state(rng, model, learning_rate=self.learning_rate, feature_dim=feature_dim)
        total_steps = self.epochs * data_loader.num_batches
        step = 0
        for _ in range(self.epochs):
            losses_cls = []
            losses_adv = []
            for batch in data_loader:
                rng, step_rng = jax.random.split(rng)
                cls_state, adv_state, loss_cls, loss_adv = train_step(
                    cls_state,
                    adv_state,
                    batch,
                    use_debias=self.use_debias,
                    adversary_loss_weight=self.adversary_loss_weight,
                    rng=step_rng,
                )
                losses_cls.append(loss_cls)
                losses_adv.append(loss_adv)
                if self.verbose > 0 and step % self.print_interval == 0:
                    adv_loss_mean = f"{np.mean(losses_adv):.6f}" if self.use_debias else None
                    logger.info(
                        f"Step {step+1}/{total_steps}: Classifier Loss = {np.mean(losses_cls):.6f}, Adversarial Loss = {adv_loss_mean}"
                    )
                step += 1
        self.cls_state, self.adv_state = cls_state, adv_state
        return self

    def _predict_proba(self, X: np.ndarray):
        inputs = jnp.array(X)
        y_prob = self.classifier.apply({"params": self.cls_state.params}, inputs, trainable=False)
        return np.array(y_prob).ravel()

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
        if not is_numeric(X):
            raise ValueError("Adversarial Debiasing only works with numeric features.")
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
        if not is_numeric(X):
            raise ValueError("Adversarial Debiasing only works with numeric features.")

        proba = np.empty((X.shape[0], 2))
        proba[:, 1] = self._predict_proba(X)
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
        if not is_numeric(X):
            raise ValueError("Adversarial Debiasing only works with numeric features.")

        p = self._predict(X).reshape([-1])
        return p
