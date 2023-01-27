import math

import numpy as np

from holisticai.bias.mitigation.inprocessing.commons import Logging


class DebiaserOptimizer:
    def __init__(self, eps0, rho, gamma, sgd_steps, batch_size, y, lambda_0=0, mu_0=0):
        self.eps0 = eps0
        self.rho = rho
        self.sgd_steps = sgd_steps
        self.lambda_k = lambda_0
        self.mu_k = mu_0
        self.batch_size = batch_size
        self.gamma = gamma
        self.y = y
        self.set_alpha()
        self.generator = self._build_generator(y)

    def get_batch(self):
        return next(self.generator)

    def compute_gradients(self, y_batch):
        lambda_minus_mu = self.lambda_k - self.mu_k
        xi_arg = np.maximum(y_batch, lambda_minus_mu)
        xi_arg = np.minimum(xi_arg, self.gamma + lambda_minus_mu)
        mean_xi = (np.mean(xi_arg) - lambda_minus_mu) / self.gamma

        lambda_gradient = self.eps0 + self.rho - mean_xi
        mu_gradient = self.eps0 - self.rho + mean_xi
        gradients = {"lambda": lambda_gradient, "mu": mu_gradient}
        return gradients

    def update_parameters(self, gradients):
        """
        If self.eps=0, we can drop mu_k and optimize lambda_k only but
        lambda_k will not be constrained to be non-negative in this case.
        """
        if self.eps0 > 1e-3:
            self.lambda_k = max(0, self.lambda_k - self.alpha * gradients["lambda"])
            self.mu_k = max(0, self.mu_k - self.alpha * gradients["mu"])
        else:
            self.lambda_k = self.lambda_k - self.alpha * gradients["lambda"]

    def update_step(self):
        y_batch = self.get_batch()
        grads = self.compute_gradients(y_batch)
        self.update_parameters(grads)


class SGDDebiaser(DebiaserOptimizer):
    def set_alpha(self):
        num_samples_sgd = self.sgd_steps * self.batch_size
        lr = self.gamma * math.sqrt(1.0 / num_samples_sgd)
        self.alpha = lr * self.batch_size

    def _build_generator(self, y):
        idx = np.arange(len(y))  # instance IDs in group k
        group_size = len(idx)
        while True:
            batch_ids = np.random.randint(0, group_size, self.batch_size)
            batch_ids = idx[batch_ids]
            yield y[batch_ids]


class FullGDDebiaser(DebiaserOptimizer):
    def set_alpha(self):
        self.alpha = 0.5

    def _build_generator(self, y):
        while True:
            yield y


class RTLogger:
    def __init__(self, total_iterations, verbose):
        self.verbose = verbose
        log_params = [("iteration", int)]
        self.logger = Logging(
            log_params=log_params,
            total_iterations=total_iterations,
            logger_format="iteration",
        )

    def update(self, iteration):
        if self.verbose:
            self.logger.update(iteration)
