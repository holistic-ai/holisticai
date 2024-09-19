import numpy as np
from holisticai.robustness.attackers.regression.gb_base import GDPoisoner
from sklearn.linear_model import Ridge


class LinRegGDPoisoner(GDPoisoner):
    """
    LinRegGDPoisoner implements computations for ordinary least squares regression. Computations involving regularization are handled in the respective children classes.

    Parameters
    ----------
    poison_proportion : float
        The proportion of points to flip. Default is 0.2.
    num_inits : int
        The number of initializations. Default is 1.
    max_iter : int
        The maximum number of iterations. Default is 15.
    eta : float
        Gradient descent step size. Default is 0.01.
    beta : float
        Decay rate for line search. Default is 0.05.
    sigma : float
        Line search stop condition. Default is 0.9.
    eps : float
        Poisoning stop condition. Default is 1e-3.
    objective : int
        Objective function to optimize. Default is 0.
    opty : bool
        Whether to optimize y. Default is True.

    References
    ----------
    .. [1] Jagielski, M., Oprea, A., Biggio, B., Liu, C., Nita-Rotaru, C., & Li, B. (2018, May). Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In 2018 IEEE symposium on security and privacy (SP) (pp. 19-35). IEEE.
    """

    def __init__(
        self,
        poison_proportion=0.2,
        num_inits=1,
        max_iter=15,
        eta=0.01,
        beta=0.05,
        sigma=0.9,
        eps=1e-3,
        objective=0,
        opty=True,
    ):
        GDPoisoner.__init__(self, eta, beta, sigma, eps, objective, opty, poison_proportion, num_inits, max_iter)
        self.initlam = 0

    def generate(self, X_train, y_train, categorical_mask=None, return_only_poisoned=False):
        """
        Generates poisoning samples.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            The training input samples.
        y_train : array-like, shape (n_samples,)
            The training target values.
        categorical_mask : array-like, shape (n_features,)
            The mask indicating whether a feature is categorical.
        return_only_poisoned : bool
            Whether to return only the poisoned samples.

        Returns
        -------
        array-like, shape (n_samples, n_features)
            The training input samples.
        array-like, shape (n_samples,)
            The training target values.
        array-like, shape (n_samples, n_features)
            The poisoning input samples.
        array-like, shape (n_samples,)
            The poisoning target values.
        """

        return self._generate(X_train, y_train, categorical_mask, return_only_poisoned)

    def learn_model(self, x, y, clf):
        """
        Trains a Ridge regression model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.
        clf : object
            The model object.

        Returns
        -------
        object
            The trained model.
        float
            The regularization parameter.
        """
        if not clf:
            clf = Ridge(alpha=0.00001)
        clf.fit(np.asarray(x), y)
        return clf, 0

    def _compute_sigma(self):
        """
        Computes the covariance matrix.

        Returns
        -------
        array-like, shape (n_features, n_features)
            The covariance matrix.
        """
        sigma = np.dot(np.transpose(self.trnx), self.trnx)
        sigma = sigma / self.trnx.shape[0]
        return sigma

    def _compute_mu(self):
        """
        Computes the mean of the input samples.

        Returns
        -------
        array-like, shape (n_features,)
            The mean of the input samples.
        """
        mu = np.mean(np.matrix(self.trnx), axis=0)
        return mu

    def _compute_m(self, clf, poisxelem, poisyelem):
        """
        Computes the matrix m.

        Parameters
        ----------
        clf : object
            The model object.
        poisxelem : array-like, shape (1, n_features)
            The poisoning input sample.
        poisyelem : float
            The poisoning target value.

        Returns
        -------
        array-like, shape (n_features, n_features)
            The matrix m.
        """
        w, b = clf.coef_, clf.intercept_
        poisxelemtransp = np.reshape(poisxelem, (self.feanum, 1))
        wtransp = np.reshape(w, (1, self.feanum))
        errterm = (np.dot(w, poisxelemtransp) + b - poisyelem).reshape((1, 1))
        first = np.dot(poisxelemtransp, wtransp)
        m = first + errterm[0, 0] * np.identity(self.feanum)
        return m

    def _compute_wb_zc(self, eq7lhs, mu, w, m, n, poisxelem):  # noqa: ARG002
        """
        Computes the weights and biases.

        Parameters
        ----------
        eq7lhs : array-like, shape (n_features+1, n_features+1)
            The left hand side of the equation.
        mu : array-like, shape (n_features,)
            The mean of the input samples.
        w : array-like, shape (n_features,)
            The weights.
        m : array-like, shape (n_features, n_features)
            The matrix m.
        n : int
            The number of samples.
        poisxelem : array-like, shape (1, n_features)
            The poisoning input sample.

        Returns
        -------
        array-like, shape (n_features, n_features)
            The weights.
        array-like, shape (n_features,)
            The biases.
        array-like, shape (n_features,)
            The weights.
        float
            The bias.
        """
        eq7rhs = -(1 / n) * np.bmat(
            [[np.matrix(m), -np.matrix(poisxelem.reshape(-1, 1))], [np.matrix(w.T), np.matrix([-1])]]
        )

        wbxc = np.linalg.lstsq(eq7lhs, eq7rhs, rcond=None)[0]
        wxc = wbxc[:-1, :-1]  # get all but last row
        bxc = wbxc[-1, :-1]  # get last row
        wyc = wbxc[:-1, -1]
        byc = wbxc[-1, -1]

        return wxc, bxc.ravel(), wyc.ravel(), byc

    def _compute_r(self, clf, lam):  # noqa: ARG002
        """
        Computes the regularization term.

        Parameters
        ----------
        clf : object
            The model object.
        lam : float
            The regularization parameter.

        Returns
        -------
        array-like, shape (1, n_features)
            The regularization term.
        """
        r = np.zeros((1, self.feanum))
        return r

    def _comp_obj_trn(self, clf, lam, otherargs):  # noqa: ARG002
        """
        Computes the objective value for the training data.

        Parameters
        ----------
        clf : object
            The model object.
        lam : float
            The regularization parameter.
        otherargs : tuple
            Other arguments.

        Returns
        -------
        float
            The objective value.
        """
        errs = clf.predict(np.asarray(self.trnx)) - self.trny
        mse = np.linalg.norm(errs) ** 2 / self.samplenum

        return mse

    def _comp_attack_trn(self, clf, wxc, bxc, wyc, byc, otherargs):  # noqa: ARG002
        """
        Computes the attack on the training data.

        Parameters
        ----------
        clf : object
            The model object.
        wxc : array-like, shape (n_features, n_features)
            The weights.
        bxc : array-like, shape (n_features,)
            The biases.
        wyc : array-like, shape (n_features,)
            The weights.
        byc : float
            The bias.
        otherargs : tuple
            Other arguments.

        Returns
        -------
        array-like, shape (n_samples, n_features)
            The attack on the input samples.
        array-like, shape (n_samples,)
            The attack on the target values.
        """
        res = clf.predict(np.asarray(self.trnx)) - self.trny

        gradx = np.dot(self.trnx, wxc) + bxc
        grady = np.dot(self.trnx, wyc.T) + byc

        attackx = np.dot(res, gradx) / self.samplenum
        attacky = np.dot(res, grady) / self.samplenum

        return attackx, attacky
