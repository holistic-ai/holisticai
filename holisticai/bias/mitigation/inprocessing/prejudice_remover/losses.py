import numpy as np


class PRBinaryCrossEntropy:
    def __init__(self, eta, C):
        self.eta = eta
        self.C = C
        self.freg = BinaryFairnessRegularizer()
        self.xent = BinaryCrossEntropy()

    def __call__(self, y, sigma, groups, coef):
        l = self.xent(y, sigma)
        f = self.freg(sigma, groups)
        # l2 regularizer
        reg = np.sum(coef * coef)
        loss = -l + self.eta * f + 0.5 * self.C * reg
        return loss

    def gradient(self, X, y, sigma, groups, coef):
        l = self.xent.gradient(X, y, sigma, groups)
        f = self.freg.gradient(X, sigma, groups)
        # l2 regularizer
        reg = coef
        # sum
        loss = np.reshape(-l + self.eta * f + self.C * reg, [-1])
        return loss


class BinaryFairnessRegularizer:
    def __call__(self, sigma, groups):
        # fairness-aware regularizer
        # \sum_{x,s in D} \
        #    sigma(x,x)       [log(rho(s))     - log(pi)    ] + \
        #    (1 - sigma(x,s)) [log(1 - rho(s)) - log(1 - pi)]
        rho_s, pi = self._parameters(sigma, groups)
        return (
            sigma * (np.log(rho_s) - np.log(pi))
            + (1.0 - sigma) * (np.log(1.0 - rho_s) - np.log(1.0 - pi))
        ).sum()

    def _parameters(self, sigma, groups):
        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        # d_rho(s) = \sum_{(xi,si)in D st si=s} d_sigma(xi,si) / #D[s]
        sigma_group = op_by_group(sigma, groups, reduce_op=np.mean)

        rho_s = sigma_group[groups]

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si) / #D
        pi = sigma.mean()

        return rho_s, pi

    def _derivate_parameters(self, X, sigma, groups):
        d_sigma = (sigma * (1.0 - sigma))[:, np.newaxis] * X

        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        # d_rho(s) = \sum_{(xi,si)in D st si=s} d_sigma(xi,si) / #D[s]
        d_sigma_group = op_by_group(d_sigma, groups, reduce_op=np.mean, axis=0)

        d_rho_s = d_sigma_group[groups]

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si) / #D
        d_pi = d_sigma.mean(axis=0)

        return d_sigma, d_rho_s, d_pi

    def gradient(self, X, sigma, groups):
        # fairness-aware regularizer
        # differentialy by w(s)
        # \sum_{x,s in {D st s=si} \
        #     [(log(rho(si)) - log(pi)) - (log(1 - rho(si)) - log(1 - pi))] \
        #     * d_sigma
        # + \sum_{x,s in {D st s=si} \
        #     [ {sigma(xi, si) - rho(si)} / {rho(si) (1 - rho(si))} ] \
        #     * d_rho
        # - \sum_{x,s in {D st s=si} \
        #     [ {sigma(xi, si) - pi} / {pi (1 - pi)} ] \
        #     * d_pi
        rho_s, pi = self._parameters(sigma, groups)
        d_sigma, d_rho_s, d_pi = self._derivate_parameters(X, sigma, groups)

        f1 = (np.log(rho_s) - np.log(pi)) - (np.log(1.0 - rho_s) - np.log(1.0 - pi))
        f2 = (sigma - rho_s) / (rho_s * (1.0 - rho_s))
        f3 = (sigma - pi) / (pi * (1.0 - pi))

        f4 = (f1[:, np.newaxis] * d_sigma + f2[:, np.newaxis] * d_rho_s) - np.outer(
            f3, d_pi
        )
        f = op_by_group(f4, groups, reduce_op=np.sum, axis=0)
        return f


class BinaryCrossEntropy:
    def __call__(self, y, sigma):
        # likelihood
        # \sum_{x,s,y in D} y log(sigma) + (1 - y) log(1 - sigma)
        return (y * np.log(sigma) + (1.0 - y) * np.log(1.0 - sigma)).sum()

    def gradient(self, X, y, sigma, groups):
        # likelihood
        # l(si) = \sum_{x,y in D st s=si} (y - sigma(x, si)) x
        loss = (y - sigma)[:, np.newaxis] * X
        l = op_by_group(loss, groups, reduce_op=np.sum, axis=0)
        return l


def op_by_group(matrix, groups, reduce_op, axis=None):
    nb_group_values = len(np.unique(groups))
    result = [reduce_op(matrix[groups == i], axis=axis) for i in range(nb_group_values)]
    return np.stack(result)
