import numpy as np
from sklearn.linear_model import LogisticRegression

EPSILON = 1.0e-10
SIGMOID_RANGE = np.log((1.0 - EPSILON) / EPSILON)


class PRLogiticRegression:
    def __init__(self, loss, initializer, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.nb_clases = 2
        self.loss = loss
        self.initializer = initializer

    def init_params(self, X, y, groups):
        # set instance variables
        X = self.preprocessing_data(X)
        self.nb_features = X.shape[1]
        self.nb_group_values = len(np.unique(groups))
        self.coef = self.initializer.initialize(X, y, groups, self.nb_group_values, self.nb_features)

    def set_params(self, coef):
        self.coef = np.reshape(coef, [self.nb_group_values, self.nb_features])

    def preprocessing_data(self, X):
        if self.fit_intercept:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def predict(self, X, groups):
        return np.argmax(self.predict_proba(X, groups), 1)

    def predict_proba(self, X, groups):
        X = self.preprocessing_data(X)
        proba = np.empty((X.shape[0], self.nb_clases))
        proba[:, 1] = self.sigmoid(X=X, groups=groups)
        proba[:, 0] = 1.0 - proba[:, 1]
        return proba

    def predict_score(self, X, groups, coef=None):
        X = self.preprocessing_data(X)
        return self.sigmoid(X=X, groups=groups, coef=coef)

    def sigmoid(self, X, groups, coef=None):
        if coef is None:
            coef = self.coef
        w = coef[groups, :]
        s = np.sum(w * X, axis=1)
        s = np.clip(s, -SIGMOID_RANGE, SIGMOID_RANGE)
        return 1.0 / (1.0 + np.exp(-s))


class PRParamInitializer:
    def __init__(self, init_type="Zero", **kargs):
        self.init_type = init_type
        if init_type.startswith("StandarLR"):
            self.C = kargs["C"]
            self.penalty = kargs["penalty"]
            self.fit_intercept = kargs["fit_intercept"]

    def initialize(self, X, y, groups, nb_group_values, nb_features):
        """set initial weight
        initialization methods are specified by `itype`
        * 0: cleared by 0
        * 1: follows standard normal distribution
        * 2: learned by standard logistic regression
        * 3: learned by standard logistic regression separately according to
            the value of sensitve feature
        Parameters
        ----------
        itype : int
            type of initialization method
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        s : array, shape=(n_samples)
            values of sensitive features
        """

        if self.init_type == "Zero":
            # clear by zeros
            coef = np.zeros(nb_group_values * nb_features, dtype=float)

        elif self.init_type == "Random":
            # at random
            coef = np.random.randn(nb_group_values * nb_features)

        elif self.init_type == "StandarLR":
            # learned by standard LR
            coef = np.empty(nb_group_values * nb_features, dtype=float)
            coef = coef.reshape(nb_group_values, nb_features)

            clr = LogisticRegression(C=self.C, penalty=self.penalty, fit_intercept=self.fit_intercept)
            clr.fit(X, y)

            coef[:, :] = clr.coef_

        elif self.init_type == "StandarLRbyGroup":
            # learned by standard LR
            coef = np.empty(nb_group_values * nb_features, dtype=float)
            coef = coef.reshape(nb_group_values, nb_features)

            for i in range(nb_group_values):
                clr = LogisticRegression(C=self.C, penalty=self.penalty, fit_intercept=self.fit_intercept)
                clr.fit(X[groups == i, :], y[groups == i])
                coef[i, :] = clr.coef_
        else:
            raise TypeError

        return coef
