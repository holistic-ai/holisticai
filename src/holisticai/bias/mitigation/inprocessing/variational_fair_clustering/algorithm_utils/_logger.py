from holisticai.bias.mitigation.inprocessing.commons import Logging


class MFLogger:
    def __init__(self, total_iterations, verbose):
        self.verbose = verbose
        log_params = [
            ("iteration", int),
            ("fairness_error", float),
            ("fair_cluster_energy", float),
            ("cluster_energy", float),
        ]
        self.total_iterations = total_iterations
        self.logger = Logging(
            log_params=log_params,
            total_iterations=self.total_iterations,
            logger_format="iteration",
        )

    def update(self, iteration, fairness_error, fair_cluster_energy, cluster_energy):
        if self.verbose:
            self.logger.update(iteration, fairness_error, fair_cluster_energy, cluster_energy)
