import numpy as np
import numpy.linalg as la
from holisticai.robustness.attackers.regression.gb_base import GDPoisoner
from sklearn import linear_model


class LinRegGDPoisoner(GDPoisoner):
    """
    LinRegGDPoisoner implements a gradient-based poisoning attack for regression models\
    by using an ordinary least squares regression model at its core.\
    The attack involves calculating gradients, selecting poison points based on these gradients, \
    assigning response values to amplify their effect, and iterating this process to generate the \
    desired number of poisoned points.

    Parameters
    ----------
    poison_proportion : float
        The proportion of points to flip. Default is 0.2.
    num_inits : int
        The number of initializations. Default is 1.
    max_iter : int
        The maximum number of iterations. Default is 15.
    initializer : str
        The initialization method. Default is 'inf_flip'.\
        Options are 'inf_flip'. 'adaptive', 'randflip' and 'randflipnobd'.
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
        initializer="inf_flip",
        eta=0.01,
        beta=0.05,
        sigma=0.9,
        eps=1e-3,
        objective=0,
        opty=True,
    ):
        GDPoisoner.__init__(
            self, eta, beta, sigma, eps, objective, opty, poison_proportion, num_inits, max_iter, initializer
        )
        self.initlam = 0

    def generate(self, X_train, y_train, categorical_mask=None, return_only_poisoned=False):
        """
        Parameters
        ----------
        X_train : pandas.DataFrame
            The training data features.
        y_train : pandas.Series
            The training data labels.
        categorical_mask : numpy.ndarray, optional
            A boolean mask indicating which columns in `X_train` are categorical.
        return_only_poisoned : bool, optional
            If True, return only the poisoned data points. Otherwise, return the entire dataset including the poisoned points.

        Returns
        -------
        pandas.DataFrame
            The features of the dataset including the poisoned points.
        pandas.Series
            The labels of the dataset including the poisoned points.

        Notes
        -----
        If `return_only_poisoned` is True, the original dataset is not modified. Otherwise, the original dataset is concatenated with the poisoned points.
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
            clf = linear_model.Ridge(alpha=0.00001)
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


class RidgeGDPoisoner(GDPoisoner):
    """
    RidgeGDPoisoner implements a gradient-based poisoning attack for regression models, \
    designed to inject malicious data points into the training dataset to manipulate the \
    model's learned parameters and degrade its predictive performance.
    Unlike `LinRegGDPoisoner`, this method includes regularization terms in the computations \
    to generate the poisoned points to maximize their impact on the regression line.

    Parameters
    ----------
    poison_proportion : float
        The proportion of points to flip. Default is 0.2.
    num_inits : int
        The number of initializations. Default is 1.
    max_iter : int
        The maximum number of iterations. Default is 15.
    initializer : str
        The initialization method. Default is 'inf_flip'.\
        Options are 'inf_flip'. 'adaptive', 'randflip' and 'randflipnobd'.
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
        initializer="inf_flip",
        eta=0.01,
        beta=0.05,
        sigma=0.9,
        eps=1e-3,
        objective=0,
        opty=True,
    ):
        GDPoisoner.__init__(
            self, eta, beta, sigma, eps, objective, opty, poison_proportion, num_inits, max_iter, initializer
        )
        self.initlam = -1

    def generate(self, X_train, y_train, categorical_mask=None, return_only_poisoned=False):
        """
        Parameters
        ----------
        X_train : pandas.DataFrame
            The training data features.
        y_train : pandas.Series
            The training data labels.
        categorical_mask : numpy.ndarray, optional
            A boolean mask indicating which columns in `X_train` are categorical.
        return_only_poisoned : bool, optional
            If True, return only the poisoned data points. Otherwise, return the entire dataset including the poisoned points.

        Returns
        -------
        pandas.DataFrame
            The features of the dataset including the poisoned points.
        pandas.Series
            The labels of the dataset including the poisoned points.

        Notes
        -----
        If `return_only_poisoned` is True, the original dataset is not modified. Otherwise, the original dataset is concatenated with the poisoned points.
        """
        self.initclf, self.initlam = self.learn_model(X_train, y_train, None, lam=None)
        return self._generate(X_train, y_train, categorical_mask, return_only_poisoned)

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
        curweight = np.linalg.norm(errs) ** 2 / self.samplenum
        l2_norm = la.norm(clf.coef_) / 2
        return lam * l2_norm + curweight

    def _comp_attack_trn(self, clf, wxc, bxc, wyc, byc, otherargs):
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
        (r,) = otherargs
        res = clf.predict(np.asarray(self.trnx)) - self.trny

        gradx = np.dot(self.trnx, wxc) + bxc
        grady = np.dot(self.trnx, wyc.T) + byc

        attackx = np.dot(res, gradx) / self.samplenum
        attacky = np.dot(res, grady) / self.samplenum

        attackx += np.dot(r, wxc)
        attacky += np.dot(r, wyc.T)
        return attackx, attacky

    def _compute_r(self, clf, lam):
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
        r += lam * np.matrix(clf.coef_).reshape(1, self.feanum)
        return r

    def _compute_sigma(self):
        """
        Computes the covariance matrix.

        Returns
        -------
        array-like, shape (n_features, n_features)
            The covariance matrix.
        """
        basesigma = np.dot(np.transpose(self.trnx), self.trnx)
        basesigma = basesigma / self.trnx.shape[0]
        sigma = basesigma + self.initlam * np.eye(self.feanum)
        return sigma

    def learn_model(self, x, y, clf, lam=None):
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
        lam = 0.1
        clf = linear_model.Ridge(alpha=lam, max_iter=10000)
        clf.fit(np.asarray(x), y)
        return clf, lam

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
