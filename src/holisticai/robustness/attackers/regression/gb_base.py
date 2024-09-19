import logging

import numpy as np
import pandas as pd
from holisticai.robustness.attackers.regression.initializers import inf_flip
from holisticai.robustness.attackers.regression.utils import one_hot_encode_columns, revert_one_hot_encoding
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


class GDPoisoner:
    """
    Gradient Descent Poisoner handles gradient poisoning routines computations for specific models found in respective classes.

    Parameters
    ----------
    eta : float
        Gradient descent step size.
    beta : float
        Decay rate for line search.
    sigma : float
        Line search stop condition.
    eps : float
        Poisoning stop condition.
    objective : int
        Objective function to optimize.
    opty : bool
        Whether to optimize y.
    poison_proportion : float
        The proportion of points to poison.
    num_inits : int
        The number of initializations. Default is 1.
    max_iter : int
        The maximum number of iterations. Default is 15.

    References
    ----------
    .. [1] Jagielski, M., Oprea, A., Biggio, B., Liu, C., Nita-Rotaru, C., & Li, B. (2018, May). Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In 2018 IEEE symposium on security and privacy (SP) (pp. 19-35). IEEE.
    """

    def __init__(self, eta, beta, sigma, eps, objective, opty, poison_proportion, num_inits=1, max_iter=15):
        self.objective = objective
        self.opty = opty

        if objective == 0:  # training MSE + regularization
            self.attack_comp = self._comp_attack_trn
            self.obj_comp = self._comp_obj_trn

        else:
            raise NotImplementedError

        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.eps = eps

        self.initclf, self.initlam = None, None

        self.colmap = None
        self.poison_proportion = poison_proportion
        self.init = inf_flip
        self.num_inits = num_inits
        self.max_iter = max_iter

    def _initialize_poison_points(self, X_train, y_train, init):
        """
        Initializes poisoning points for the poisoning routine. It uses the initialization method specified in the input, and returns the poisoning points.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            The input samples.
        y_train : array-like, shape (n_samples,)
            The target values.
        init : callable
            The initialization method.

        Returns
        -------
        poisx : array-like, shape (n_poison_samples, n_features)
            The poisoning input samples.
        poisy : array-like, shape (n_poison_samples,)
            The poisoning target values.
        """
        besterr = -1

        for _ in range(self.num_inits):
            poisx, poisy = init(X_train, y_train, self.poison_proportion)
            x_data = np.concatenate((X_train, poisx), axis=0)
            y_data = np.concatenate([y_train, poisy])
            clf = Ridge(alpha=0.00001)
            clf.fit(np.asarray(x_data), y_data)
            err = mean_squared_error(y_train, clf.predict(X_train))
            logger.info("Training Error: %f", err)
            if err > besterr:
                bestpoisx, bestpoisy, besterr = np.copy(poisx), poisy[:], err
        poisx, poisy = bestpoisx, bestpoisy
        logger.info("Best initialization error: %f", besterr)

        return poisx, poisy

    def _generate(self, X_train, y_train, categorical_mask=None, return_only_poisoned=False):
        """
        Takes an initial set of poisoning points and optimizes it using gradient descent.

        Parameters
        ----------
        X_train : Dataframe, shape (n_samples, n_features)
            The input samples.
        y_train : Series, shape (n_samples,)
            The target values.
        categorical_mask : array-like, shape (n_features,)
            The mask for categorical features.
        return_only_poisoned : bool
            Whether to return only the poisoned samples.
        Returns
        -------
        Dataframe, shape (n_samples, n_features)
            The poisoned input samples.
        Series, shape (n_samples,)
            The poisoned target values.
        """

        X_original = X_train.copy()
        y_original = y_train.copy()

        x_columns = X_train.columns
        y_column = y_train.name

        if categorical_mask is not None:
            categorical_columns = X_train.columns[categorical_mask.astype(bool)].to_list()
            column_mapping, X_train = one_hot_encode_columns(X_train, categorical_columns)
            x_oh_columns = X_train.columns
            self.colmap = column_mapping

        poisx, poisy = self._initialize_poison_points(X_train, y_train, self.init)
        self.trnx, self.trny = X_train, y_train
        self.samplenum = X_train.shape[0]
        self.feanum = X_train.shape[1]

        poisct = poisx.shape[0]
        logger.info("Poison Count: %f", poisct)

        new_poisx = np.zeros(poisx.shape)
        new_poisy = [None for a in poisy]

        best_poisx = np.zeros(poisx.shape)
        best_poisy = [None for a in poisy]

        best_obj = 0
        count = 0

        sig = self._compute_sigma()  # can already compute sigma and mu
        mu = self._compute_mu()  # as x_c does not change them
        eq7lhs = np.bmat([[sig, np.transpose(mu)], [mu, np.matrix([1])]])

        # initial model - used in visualization
        clf_init, lam_init = self.learn_model(self.trnx, self.trny, None)
        clf, lam = clf_init, lam_init

        # figure out starting error
        it_res = self.iter_progress(poisx, poisy, poisx, poisy)

        logger.info(f"Iteration {count}:")
        logger.info(f"Objective Value: {it_res[0]} Change: {it_res[0]}")

        if it_res[0] > best_obj:
            best_poisx, best_poisy, best_obj = poisx, poisy, it_res[0]

        # main work loop
        while True:
            count += 1
            new_poisx = np.matrix(np.zeros(poisx.shape))
            new_poisy = [None for a in poisy]
            x_cur = np.concatenate((self.trnx, poisx), axis=0)
            y_cur = np.concatenate([self.trny, poisy])

            clf, lam = self.learn_model(x_cur, y_cur, None)
            pois_params = [(poisx[i], poisy[i], eq7lhs, mu, clf, lam) for i in range(poisct)]
            outofboundsct = 0

            for i in range(poisct):
                cur_pois_res = self.poison_data_subroutine(pois_params[i])

                new_poisx[i] = cur_pois_res[0]
                new_poisy[i] = cur_pois_res[1]
                outofboundsct += cur_pois_res[2]

            it_res = self.iter_progress(poisx, poisy, new_poisx, new_poisy)

            logger.info(f"Iteration {count}:")
            logger.info(f"Objective Value: {it_res[0]} Change: {it_res[0] - it_res[1]}")

            logger.info(f"Y pushed out of bounds: {outofboundsct}/{poisct}")

            # if we don't make progress, decrease learning rate
            if it_res[0] < it_res[1]:
                logger.info("no progress")
                self.eta *= 0.75
                new_poisx, new_poisy = poisx, poisy
            else:
                poisx = new_poisx
                poisy = new_poisy

            if it_res[0] > best_obj:
                best_poisx, best_poisy, best_obj = poisx, poisy, it_res[1]

            it_diff = abs(it_res[0] - it_res[1])
            # stopping conditions
            if count >= self.max_iter and (it_diff <= self.eps) or count >= self.max_iter:
                break

        self.best_poisx = best_poisx
        self.best_poisy = best_poisy

        if categorical_mask is not None:
            best_poisx = revert_one_hot_encoding(best_poisx, column_mapping, x_oh_columns)
        else:
            best_poisx = pd.DataFrame(best_poisx, columns=x_columns)
        best_poisy = pd.Series(best_poisy, name=y_column)
        if return_only_poisoned:
            return best_poisx, best_poisy
        x_out = np.concatenate((X_original, best_poisx), axis=0)
        y_out = np.concatenate([y_original, best_poisy])

        return pd.DataFrame(x_out, columns=x_columns), pd.Series(y_out, name=y_column)

    def poison_data_subroutine(self, in_tuple):
        """
        Poisons a single poisoning point input is passed in as a tuple and immediately unpacked.

        Parameters
        ----------
        in_tuple : tuple
            Tuple containing poisoning point, poisoning target, eq7lhs, mu, clf, lam.

        Returns
        -------
        poisxelem : array-like, shape (1, n_features)
            The poisoned input sample.
        poisyelem : float
            The poisoned target value.
        outofbounds : bool
            Whether the target value was pushed out of bounds.
        """

        poisxelem, poisyelem, eq7lhs, mu, clf, lam = in_tuple
        m = self._compute_m(clf, poisxelem, poisyelem)

        # compute partials
        wxc, bxc, wyc, byc = self._compute_wb_zc(eq7lhs, mu, clf.coef_, m, self.samplenum, poisxelem)

        if self.objective == 0:
            r = self._compute_r(clf, lam)
            otherargs = (r,)
        else:
            otherargs = None

        attack, attacky = self.attack_comp(clf, wxc, bxc, wyc, byc, otherargs)

        # keep track of how many points are pushed out of bounds
        outofbounds = bool(poisyelem >= 1 and attacky >= 0 or poisyelem <= 0 and attacky <= 0)

        # include y in gradient normalization
        if self.opty:
            allattack = np.array(np.concatenate((attack, attacky), axis=1))
            allattack = allattack.ravel()
        else:
            allattack = attack.ravel()

        norm = np.linalg.norm(allattack)
        allattack = allattack / norm if norm > 0 else allattack
        if self.opty:
            attack, attacky = allattack[:-1], allattack[-1]
        else:
            attack = allattack

        poisxelem, poisyelem, _ = self.linesearch(poisxelem, poisyelem, attack, attacky)
        poisxelem = poisxelem.reshape((1, self.feanum))

        return poisxelem, poisyelem, outofbounds

    def linesearch(self, poisxelem, poisyelem, attack, attacky):
        """
        Line search routine for poisoning points.

        Parameters
        ----------
        poisxelem : array-like, shape (1, n_features)
            The poisoning input sample.
        poisyelem : float
            The poisoning target value.
        attack : array-like, shape (1, n_features)
            The attack on the input sample.
        attacky : float
            The attack on the target value.

        Returns
        -------
        array-like, shape (1, n_features)
            The poisoned input sample.
        float
            The poisoned target value.
        float
            The objective value.
        """
        k = 0
        x0 = np.copy(self.trnx)
        y0 = self.trny[:]

        curx = np.append(x0, poisxelem.reshape(1, -1), axis=0)
        cury = y0[:]
        cury = np.append(cury, poisyelem)

        clf, lam = self.learn_model(curx, cury, None)
        clf1, lam1 = clf, lam

        lastpoisxelem = poisxelem
        curpoisxelem = poisxelem

        lastyc = poisyelem
        curyc = poisyelem
        otherargs = None

        w_1 = self.obj_comp(clf, lam, otherargs)
        count = 0
        eta = self.eta

        while True:
            if count > 0:
                eta = self.beta * eta
            count += 1
            curpoisxelem = curpoisxelem + eta * attack
            curpoisxelem = np.clip(curpoisxelem, 0, 1)
            curx[-1] = curpoisxelem

            if self.opty:
                curyc = curyc + attacky * eta
                curyc = min(1, max(0, curyc))
                cury[-1] = curyc
            clf1, lam1 = self.learn_model(curx, cury, clf1)
            w_2 = self.obj_comp(clf1, lam1, otherargs)

            if count >= 100 or abs(w_1 - w_2) < 1e-8:  # convergence
                break
            if w_2 - w_1 < 0:  # bad progress
                curpoisxelem = lastpoisxelem
                curyc = lastyc
                break

            lastpoisxelem = curpoisxelem
            lastyc = curyc
            w_1 = w_2
            k += 1

        curpoisxelem = curpoisxelem.reshape(1, -1)

        if self.colmap is not None:
            for col in self.colmap:
                vals = [(curpoisxelem[0, j], j) for j in self.colmap[col]]
                topval, topcol = max(vals)
                for j in self.colmap[col]:
                    if j != topcol:
                        curpoisxelem[0, j] = 0
                if topval > 1 / (1 + len(self.colmap[col])):
                    curpoisxelem[0, topcol] = 1
                else:
                    curpoisxelem[0, topcol] = 0
        curx = np.delete(curx, curx.shape[0] - 1, axis=0)
        curx = np.append(curx, curpoisxelem.reshape(1, -1), axis=0)
        cury[-1] = curyc
        clf1, lam1 = self.learn_model(curx, cury, None)

        w_2 = self.obj_comp(clf1, lam1, otherargs)

        return np.clip(curpoisxelem, 0, 1), curyc, w_2

    def iter_progress(self, lastpoisx, lastpoisy, curpoisx, curpoisy):
        """
        Computes the objective value for the poisoning points.

        Parameters
        ----------
        lastpoisx : array-like, shape (n_poison_samples, n_features)
            The last poisoning input samples.
        lastpoisy : array-like, shape (n_poison_samples,)
            The last poisoning target values.
        curpoisx : array-like, shape (n_poison_samples, n_features)
            The current poisoning input samples.
        curpoisy : array-like, shape (n_poison_samples,)
            The current poisoning target values.

        Returns
        -------
        float
            The objective value.
        float
            The previous objective value.
        """
        x0 = np.concatenate((self.trnx, lastpoisx), axis=0)
        y0 = np.concatenate([self.trny, lastpoisy])
        clf0, lam0 = self.learn_model(x0, y0, None)
        w_0 = self.obj_comp(clf0, lam0, None)

        x1 = np.concatenate((self.trnx, curpoisx), axis=0)
        y1 = np.concatenate([self.trny, curpoisy])
        clf1, lam1 = self.learn_model(x1, y1, None)
        w_1 = self.obj_comp(clf1, lam1, None)

        return w_1, w_0
