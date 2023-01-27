from functools import partial

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

from holisticai.utils.transformers.bias import SensitiveGroups


def prob(dist, x):
    return dist.pdf(x)


class MetaFairClassifierAlgorithm:
    def __init__(self, logger, tau, eps=0.01, batch_size=20, constraint=None, steps=10):
        self.tau = tau
        self.eps = eps
        self.batch_size = batch_size
        self.steps = steps
        self.constraint = constraint
        self.logger = logger
        self.sens_groups = SensitiveGroups()

    def range(self, eps, tau):
        a = np.arange(np.ceil(tau / eps), step=self.steps) * eps
        b = (a + eps) / tau
        b = np.minimum(b, 1)
        return np.c_[a, b]

    def probabilities(self, dist, x):
        pos = np.ones(len(x))
        zero = np.zeros(len(x))

        P = {}

        P["Y=1,Z=1"] = prob(dist, np.c_[x, pos, pos])
        P["Y=-1,Z=1"] = prob(dist, np.c_[x, -pos, pos])
        P["Y=1,Z=0"] = prob(dist, np.c_[x, pos, zero])
        P["Y=-1,Z=0"] = prob(dist, np.c_[x, -pos, zero])

        P["total"] = P["Y=1,Z=1"] + P["Y=1,Z=0"] + P["Y=-1,Z=0"] + P["Y=-1,Z=1"]

        P["Y=1"] = (P["Y=1,Z=1"] + P["Y=1,Z=0"]) / P["total"]
        P["Z=0"] = (P["Y=-1,Z=0"] + P["Y=1,Z=0"]) / P["total"]
        P["Z=1"] = (P["Y=-1,Z=1"] + P["Y=1,Z=1"]) / P["total"]

        return P

    def compute_gradient(self, P, a, b):
        """Gradient Descent implementation for the optimizing the objective
        function.

        Note that one can alternately also use packages like CVXPY here.
        Here we use decaying step size. For certain objectives, constant step
        size might be better.
        """

        def init_params(i):
            return [i - 5] * self.constraint.num_params

        min_cost = np.inf  # 1e8
        best_param = None
        for i in range(1, 10):
            params = init_params(i)
            for k in range(1, 50):
                grad = self.constraint.expected_gradient(P, params, a, b)
                for j in range(self.constraint.num_params):
                    params[j] = params[j] - 1 / k * grad[j]

                cost = self.constraint.cost_function(P, params, a, b)
                # print(cost)
                if cost < min_cost:
                    min_cost, best_param = cost, params
        return best_param

    def forward(self, dist, params, a, b, x):
        P = self.probabilities(dist=dist, x=x)
        return self.constraint.forward(P=P, params=params, a=a, b=b)

    def fit(self, X, y, sensitive_features, random_state=None):
        y_true = y.copy()
        y_true[y == 0] = -1

        groups_num = self.sens_groups.fit_transform(
            sensitive_features, convert_numeric=True
        )

        """Returns the model given the training data and input tau."""
        train = np.concatenate(
            [X, y_true[:, np.newaxis], groups_num[:, np.newaxis]], axis=1
        )
        mean = np.mean(train, axis=0)
        cov = np.cov(train, rowvar=False)
        dist = multivariate_normal(mean, cov, allow_singular=True, seed=random_state)
        nb_features = X.shape[1]
        dist_x = multivariate_normal(
            mean[:nb_features],
            cov[:nb_features, :nb_features],
            allow_singular=True,
            seed=random_state,
        )

        z_prior = np.mean(groups_num)
        self.constraint.z_prior = z_prior
        max_acc = -np.inf
        params_opt = [0] * self.constraint.num_params
        p, q = 0, 0
        for it, (a, b) in enumerate(self.range(self.eps, self.tau), 1):
            samples = dist_x.rvs(size=self.batch_size)

            P = self.probabilities(dist, samples)

            params = self.compute_gradient(P, a, b)

            t = self.forward(dist, params, a, b, X)

            y_pred = np.where(t > 0, 1, -1)

            acc = accuracy_score(y_true, y_pred)

            gamma = self.constraint.gamma(y_true, y_pred, groups_num)

            if max_acc < acc and gamma >= self.tau - 0.2:  # TODO: why - 0.2?
                max_acc = acc
                params_opt = params
                p, q = a, b

            self.logger.update(it, max_acc, gamma)

        self.predictor = partial(self.forward, dist, params_opt, p, q)
        return self

    def predict(self, X: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """

        t = self.predictor(X)
        y = (t > 0).astype(int).reshape((-1, 1))
        return y.ravel()

    def predict_proba(self, X: np.ndarray):
        """
        Probability Prediction

        Description
        ----------
        Probability estimate for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        Returns
        -------
        numpy.ndarray
            probability output per sample.
        """
        t = self.predictor(X)
        scores = ((t + 1) / 2).reshape((-1, 1))
        return scores
