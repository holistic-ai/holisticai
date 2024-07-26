import copy

import numpy as np
from holisticai.bias.mitigation.postprocessing.ml_debiaser.randomized_threshold.algorithm import (
    RandomizedThresholdAlgorithm,
)
from holisticai.bias.mitigation.postprocessing.ml_debiaser.reduce2binary.algorithm_utils import RBLogger


class Reduce2BinaryAlgorithm:
    """
    Debiase multiclass datasets via preprocessing (R2B Algorithm).
    Use Alternating Direction Method of multipliers (ADMM) to decompose the problem into separate
    debiasing tasks of binary labels before they are aggregated.
    """

    def __init__(
        self,
        gamma=1.0,
        eps=0,
        eta=0.5,
        num_classes=2,
        sgd_steps=10,
        full_gradient_epochs=1_000,
        batch_size=256,
        max_iter=100,
        verbose=True,
    ):
        """
        Instantiate object.

        Parameters
        ----------
          gamma: float
            The regularization parameter gamma (for randomization). Set this to
            1 if the goal is to minimize changes to the original probability scores.

          eps: float
            Tolerance parameter for bias >= 0.

          eta: float
            The step size parameters in ADMM.

          num_classes: int
            Number of classes (must be >= 2).

          sgd_steps: int
            Number of minibatch steps in SGD.

          full_gradient_epochs: int
            Number of full gradient descent steps.

          batch_size: int
            Size of minibatches in SGD.

          max_iter: int
            Maximum number of iteration of the ADMM procedure.

          verbose: bool
            If True, display progress.
        """
        if num_classes < 2:
            raise ValueError("Number of classes (must be >= 2).")

        if eps < 0:
            raise ValueError("eps must be non-negative.")

        if gamma <= 0:
            raise ValueError("gamma must be a strictly positive number.")

        self.num_groups = 1
        self.gamma = gamma
        self.eps = eps
        self.eta = eta
        self.num_classes = num_classes
        self.max_iter = max_iter

        # binary debiasers for each label
        self.debiasers = {}
        for k in range(num_classes):
            self.debiasers[k] = RandomizedThresholdAlgorithm(
                gamma=gamma + eta,
                eps=eps,
                sgd_steps=sgd_steps,
                full_gradient_epochs=full_gradient_epochs,
                batch_size=batch_size,
                verbose=False,
            )

        self.logger = RBLogger(verbose=verbose, total_iterations=max_iter)

    def _compute_z(self, h_mat, u_mat):
        # Compute the Z matrix in the R2B algorithm.
        mult_by_ones = np.matmul(
            h_mat + u_mat,
            np.ones(
                self.num_classes,
            ),
        )
        over_k = 1.0 / self.num_classes * (mult_by_ones - np.ones(mult_by_ones.shape))
        j_mat = np.outer(
            over_k,
            np.ones(
                self.num_classes,
            ),
        )
        return h_mat + u_mat - j_mat

    def predict(self, y_prob, p_attr):
        """
        Description
        -----------
        Debias scores w.r.t. the sensitive class in each demographic group.
        In the multiclass setting, we use ADMM to decompose the problem into
        separate debiasing tasks of binary labels before they are aggregated.

        Parameters
        ----------
          y_prob: np.ndarray
            Predicted probability matrix.

          p_attr: An array containing the group id of each instance starting
            from group 0 to group (num_classes-1).

        Returns:
          fair infered probability matrix.
        """

        if len(y_prob.shape) != 2:
            raise ValueError(
                "Original prob scores must be a 2-dimensional array."
                "Use RandomizedThreshold for binary classification."
            )

        y_prob_scores = copy.deepcopy(y_prob)

        # Initialize ADMM.
        f_mat = copy.deepcopy(y_prob_scores)
        h_mat = np.zeros_like(f_mat)
        u_mat = np.zeros_like(f_mat)
        z_mat = np.zeros_like(f_mat)

        for iteration in range(self.max_iter):
            # Step 1: debias each label separately.
            for k in range(self.num_classes):
                self.debiasers[k].fit(f_mat[:, k], p_attr)

                h_mat[:, k] = self.debiasers[k].predict(f_mat[:, k], p_attr)

                # Step 2: update ADMM variables.
                old_z = copy.deepcopy(z_mat)
                z_mat = self._compute_z(h_mat, u_mat)
                u_mat = u_mat + h_mat - z_mat
                f_mat = y_prob + self.eta * (z_mat - u_mat)

                # Compute primal and dual residuals.
                s = np.linalg.norm(z_mat - old_z)
                r = np.linalg.norm(z_mat - h_mat)

            self.logger.update(iteration=iteration + 1, primal_residual=r, dual_residual=s)

        z_mat = np.maximum(z_mat, 0)
        z_mat = z_mat / np.sum(z_mat, axis=1, keepdims=True)
        return z_mat
