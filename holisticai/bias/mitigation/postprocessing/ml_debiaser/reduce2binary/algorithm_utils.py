from holisticai.bias.mitigation.inprocessing.commons import Logging


class RBLogger:
    def __init__(self, total_iterations, verbose):
        self.verbose = verbose
        log_params = [
            ("iteration", int),
            ("primal_residual:", float),
            ("dual_residual:", float),
        ]
        self.logger = Logging(
            log_params=log_params,
            total_iterations=total_iterations,
            logger_format="iteration",
        )

    def update(self, iteration, primal_residual, dual_residual):
        if self.verbose:
            self.logger.update(iteration, primal_residual, dual_residual)
