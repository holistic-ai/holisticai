import numpy as np
from holisticai.bias.mitigation.inprocessing.commons import Logging


class MFLogger:
    def __init__(self, tau, eps, steps, verbose):
        self.verbose = verbose
        log_params = [("iteration", int), ("accuracy", float), ("gamma", float)]
        self.total_iterations = np.ceil(tau / eps) // steps
        self.logger = Logging(
            log_params=log_params,
            total_iterations=self.total_iterations,
            logger_format="iteration",
        )

    def update(self, it, max_acc, gamma):
        if self.verbose:
            self.logger.update(it, max_acc, gamma)
