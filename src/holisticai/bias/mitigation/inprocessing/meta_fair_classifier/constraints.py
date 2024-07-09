import numpy as np


class StatisticalRate:
    def __init__(self, mu):
        self.mu = mu
        self.num_params = 2

    def gamma(self, y_true, y_pred, groups_num):  # noqa: ARG002
        pos_0 = np.mean(y_pred[groups_num == 0] == 1)
        pos_1 = np.mean(y_pred[groups_num == 1] == 1)
        if pos_0 == 0 or pos_1 == 0:
            return 0
        return min(pos_0 / pos_1, pos_1 / pos_0)

    def forward(self, P, params, a=None, b=None, return_cs=False):  # noqa: ARG002
        l_1, l_2 = params
        z_0, z_1 = 1 - self.z_prior, self.z_prior

        c_0 = P["Y=1"] - 0.5
        c_1 = P["Z=0"] / z_0
        c_2 = P["Z=1"] / z_1

        t = c_0 + c_1 * l_1 + c_2 * l_2

        if return_cs:
            return t, c_1, c_2
        return t

    def _gradient(self, a, b, t, c, l):
        _t = t * c / np.sqrt(t**2 + self.mu**2)
        exp = np.mean(_t)
        dl = exp - b + (b - a) / 2 + (b - a) * l / (2 * np.sqrt(l**2 + self.mu**2))
        return dl

    def expected_gradient(self, P, params, a, b):
        l_1, l_2 = params
        t, c_1, c_2 = self.forward(P, params, return_cs=True)
        dl1 = self._gradient(a, b, t, c_1, l_1)
        dl2 = self._gradient(a, b, t, c_2, l_2)
        return dl1, dl2

    def cost_function(self, P, params, a, b):
        l_1, l_2 = params

        exp = np.mean(np.abs(self.forward(P, params)))
        result = exp - b * l_1 + -b * l_2
        if l_1 > 0:
            result += (b - a) * l_1
        if l_2 > 0:
            result += (b - a) * l_2

        return result


class FalseDiscovery:
    def __init__(self, mu):
        self.mu = mu
        self.num_params = 4

    def gamma(self, y_true, y_pred, groups_num):
        pos_0 = y_pred[groups_num == 0] == 1
        pos_1 = y_pred[groups_num == 1] == 1
        if np.sum(pos_0) == 0 or np.sum(pos_1) == 0:
            return 0
        fdr_0 = np.sum(pos_0 & (y_true[groups_num == 0] == -1)) / np.sum(pos_0)
        fdr_1 = np.sum(pos_1 & (y_true[groups_num == 1] == -1)) / np.sum(pos_1)
        if fdr_0 == 0 or fdr_1 == 0:
            return 0
        return min(fdr_0 / fdr_1, fdr_1 / fdr_0)

    def forward(self, P, params, a, b, return_probs=False):
        P["Y=-1|Z=0"] = P["Y=-1,Z=0"] / P["total"]
        P["Y=-1|Z=1"] = P["Y=-1,Z=1"] / P["total"]

        u_1, u_2, l_1, l_2 = params
        c_0 = P["Y=1"] - 0.5
        c_1 = u_1 * (P["Y=-1|Z=0"] - a * P["Z=0"]) + u_2 * (P["Y=-1|Z=1"] - a * P["Z=1"])
        c_2 = l_1 * (-P["Y=-1|Z=0"] + b * P["Z=0"]) + l_2 * (-P["Y=-1|Z=1"] + b * P["Z=1"])

        t = c_0 + c_1 * l_1 + c_2 * l_2

        if return_probs:
            return t, P
        return t

    def expected_gradient(self, P, params, a, b):
        t, P = self.forward(P, params, a, b, return_probs=True)
        res = np.vstack(
            [
                P["Y=-1|Z=0"] - a * P["Z=0"],
                P["Y=-1|Z=1"] - a * P["Z=1"],
                -P["Y=-1|Z=0"] + b * P["Z=0"],
                -P["Y=-1|Z=1"] + b * P["Z=1"],
            ]
        )
        res *= t / np.sqrt(t**2 + self.mu**2)
        return np.mean(res, axis=1)

    def cost_function(self, P, params, a, b):
        return np.mean(np.abs(self.forward(P, params, a, b)))
