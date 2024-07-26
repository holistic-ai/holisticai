import numpy as np
from holisticai.bias.mitigation.inprocessing.commons import Logging


class ObjectiveFunction:
    """objective function of logistic regression with prejudice remover
    Loss Function type 4: Weights for logistic regression are prepared for each
    value of S. Penalty for enhancing is defined as mutual information between
    Y and S.
    """

    def __init__(self, estimator, loss_fn):
        self.loss_fn = loss_fn
        self.estimator = estimator

    def loss(self, coef_, X, y, groups):
        """loss function: negative log - likelihood with l2 regularizer
        To suppress the warnings at np.log, do "np.seterr(all='ignore')"
        Parameters
        ----------
        `coef_` : array, shape=(`nb_group_values` * n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        sensitive_features : array, shape=(n_samples)
            values of sensitive features
        Returns
        -------
        loss : float
            loss function value
        """
        coef = coef_.reshape(self.estimator.nb_group_values, self.estimator.nb_features).astype(np.float64)
        X = self.estimator.preprocessing_data(X)
        sigma = self.estimator.sigmoid(X=X, groups=groups, coef=coef)
        loss = self.loss_fn(y=y, sigma=sigma, groups=groups, coef=coef)
        return loss

    def grad_loss(self, coef_, X, y, groups):
        """first derivative of loss function
        Parameters
        ----------
        `coef_` : array, shape=(`nb_group_values` * n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        s : array, shape=(n_samples)
            values of sensitive features
        Returns
        grad_loss : float
            first derivative of loss function
        """

        coef = coef_.reshape(self.estimator.nb_group_values, self.estimator.nb_features).astype(np.float64)
        X = self.estimator.preprocessing_data(X)
        sigma = self.estimator.sigmoid(X=X, groups=groups, coef=coef)
        return self.loss_fn.gradient(X=X, y=y, sigma=sigma, groups=groups, coef=coef)


class PRLogger:
    def __init__(self, total_iterations, verbose, print_interval):
        self.step = 0
        self.verbose = verbose
        self.print_interval = print_interval
        self.total_iterations = total_iterations
        self.log_params = [
            ("iteration", int),
        ]
        self.fn_params = {}
        self.logger = Logging(
            log_params=self.log_params,
            total_iterations=self.total_iterations,
            logger_format="iteration",
        )

    def set_log_fn(self, htype, **kargs):
        for param_name, param_fn in kargs.items():
            self.log_params.append((param_name, htype))
            self.fn_params[param_name] = param_fn

    def callback(self, x):
        self.step += 1
        if self.verbose > 0:  # noqa: SIM102
            if (self.step % self.print_interval) == 0 or (self.step % self.total_iterations) == 0:
                args = [self.fn_params[pn](x) for pn, _ in self.log_params[1:]]
                self.logger.update(self.step, *args)

    def info(self, message):
        if self.verbose > 0:
            self.logger.info(message)
