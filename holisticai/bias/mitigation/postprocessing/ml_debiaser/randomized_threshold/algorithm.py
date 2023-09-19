import numpy as np

from .algorithm_utils import FullGDDebiaser, RTLogger, SGDDebiaser


class RandomizedThresholdAlgorithm:
    """
    Threshold optimizer (RTO) to debias models via postprocessing.
    Debias predictions w.r.t. the sensitive class in each demographic group.
    This procedure takes as input a vector y and solves the optimization
    problem subject to the statistical parity constraint.

    Reference
    ---------
    Alabdulmohsin, Ibrahim M., and Mario Lucic. "A near-optimal algorithm for debiasing
    trained machine learning models." Advances in Neural Information Processing Systems
    34 (2021): 8072-8084.
    """

    def __init__(
        self,
        gamma=1.0,
        eps=0.0,
        rho=None,
        sgd_steps=10_000,
        full_gradient_epochs=1_000,
        batch_size=256,
        verbose=False,
    ):
        """
        Instantiate object.

        Parameters
        ----------
          gamma: float
            The regularization parameter gamma (for randomization). Set this to
            1 if the goal is to minmize changes to the original scores.

          eps: float
            Tolerance parameter for bias between 0 and 1 inclusive.

          rho: float
            The rho parameter in the post-hoc rule. If None, rho = E[y].

          sgd_steps: int
            Number of minibatch steps in SGD.

          full_gradient_epochs: int
            Number of epochs in full gradient descent phase.

          batch_size: int
            Size of minibatches in SGD.

          verbose: bool
            If True, display progress.
        """
        self.num_groups = 1
        self.gamma = gamma
        self.eps = eps
        self.rho = rho
        self.sgd_steps = sgd_steps
        self.full_gradient_epochs = full_gradient_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, y_score, groups_num):
        """
        Run the debiaser optimzier

        Description
        -----------
          minimize_x   gamma/2 ||x||^2 - y^Tx
          s.t.      x satisfies DP constraints with tolerance eps and parameter rho.
        The optimizer proceeds in two rounds:
          - First is SGD.
          - Second is full gradient descent.

        Parameters
        ----------
          y_score: np.ndarray
            Predicted probability vector

          groups_num: np.ndarray
            Group membership vector from group 0 to group (num_classes-1).
        Returns:
          None.
        """
        self.yscale = "positive" if min(y_score) >= 0 else "negative"
        self.avrg_y_score = float(sum(y_score)) / len(y_score)
        if self.rho is None:
            if self.yscale == "positive":
                self.rho = self.avrg_y_score
            else:
                self.rho = self.avrg_y_score / 2.0 + 0.5

        if self.rho <= 0:
            raise ValueError("rho must be either None or a strictly positive number.")

        num_groups = len(set(groups_num))
        eps0 = self.eps / 2.0

        kargs = {
            "eps0": eps0,
            "rho": self.rho,
            "gamma": self.gamma,
            "sgd_steps": self.sgd_steps,
            "batch_size": self.batch_size,
        }

        self.lambdas = np.zeros((num_groups,))
        self.mus = np.zeros((num_groups,))
        logger = RTLogger(total_iterations=num_groups, verbose=self.verbose)
        for k in range(num_groups):
            group_k = np.where(groups_num == k)
            optimizer = SGDDebiaser(y=y_score[group_k], **kargs)

            # Step 1: SGD
            lambda_final = 0
            mu_final = 0
            for _ in range(self.sgd_steps):
                optimizer.update_step()
                lambda_final += optimizer.lambda_k / self.sgd_steps
                mu_final += optimizer.mu_k / self.sgd_steps

            # Step 2: Full gradient descent
            optimizer = FullGDDebiaser(
                y=y_score[group_k], lambda_0=lambda_final, mu_0=mu_final, **kargs
            )
            for _ in range(self.full_gradient_epochs):
                optimizer.update_step()

            self.lambdas[k] = optimizer.lambda_k
            self.mus[k] = optimizer.mu_k

            logger.update(iteration=k + 1)

    def predict(self, y_score, p_attr):
        """
        Debiases the predictions.

        Description
        -----------

        Given the original scores y, post-process them according to the learned
        model such that the predictions satisfy the desired fairness criteria.

        Parameters
        ----------
          y_orig: np.ndarray
            Predicted probability Vector

          groups_num: np.ndarray
            Group membership vector from group 0 to group (num_classes-1).

        Returns
        -------
          y_score: is the probability of predicting the positive
            class for the instance i.
        """
        num_examples = len(y_score)
        gamma = self.gamma
        lambdas = self.lambdas
        mus = self.mus

        new_y_score = np.zeros((num_examples,))

        for i, k in enumerate(p_attr):
            if y_score[i] < (lambdas[k] - mus[k]):
                new_y_score[i] = 0
            elif y_score[i] < (lambdas[k] - mus[k]) + gamma:
                new_y_score[i] = (1.0 / gamma) * (y_score[i] - (lambdas[k] - mus[k]))
            else:
                new_y_score[i] = 1.0

        return new_y_score
