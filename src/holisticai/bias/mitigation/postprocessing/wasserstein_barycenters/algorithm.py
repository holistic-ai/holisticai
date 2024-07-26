import numpy as np
from holisticai.utils.transformers.bias import SensitiveGroups


class WassersteinBarycenterAlgorithm:
    def __init__(self):
        self._sensgroups = SensitiveGroups()
        self.eps = np.finfo(float).eps

    def fit(self, y_pred: np.ndarray, sensitive_groups: np.ndarray):
        p_attr = self._sensgroups.fit_transform(sensitive_groups, convert_numeric=True).squeeze()
        self.group_values = np.unique(p_attr)

        group_freq = [
            np.sum(p_attr == self.group_values[0]),
            np.sum(p_attr == self.group_values[1]),
        ]

        self.iM = np.argmax(group_freq)
        self.nM = group_freq[self.iM]

        self.im = np.argmin(group_freq)
        self.nm = group_freq[self.im]

        self.p = self.nm / len(p_attr)
        self.q = 1 - self.p

        self.SL = p_attr

        noise = self.eps * np.random.randn(len(y_pred)).squeeze()
        self.YL = y_pred + noise

        self.minY = np.min(self.YL)
        self.maxY = np.max(self.YL)

    def _optimize_ts(self, yt, i1, i2, n1, n2):
        mask1 = self.group_values[i1] == self.SL
        mask2 = self.group_values[i2] == self.SL
        yl_masked2 = self.YL[mask2]
        tmp2 = np.sum(yl_masked2 < yt) / n2
        yl_masked1 = self.YL[mask1]
        t_values = np.linspace(self.minY, self.maxY, 100)
        tmp1 = np.sum(yl_masked1[:, None] < t_values, axis=0) / n1
        dist = np.abs(tmp1 - tmp2)
        ts = t_values[np.argmin(dist)]
        return ts

    def _update_yt(self, yt, group):
        if group == self.group_values[self.im]:
            ts = self._optimize_ts(yt, self.iM, self.im, self.nM, self.nm)
            yf = self.p * yt + (1 - self.p) * ts

        else:
            ts = self._optimize_ts(yt, self.im, self.iM, self.nm, self.nM)
            yf = self.q * yt + (1 - self.q) * ts
        return yf

    def transform(self, y_pred: np.ndarray, sensitive_groups: np.ndarray):
        ST = self._sensgroups.transform(sensitive_groups, convert_numeric=True)
        noise = self.eps * np.random.randn(len(y_pred)).squeeze()
        YT = y_pred + noise
        YF = np.array([self._update_yt(yt, st) for yt, st in zip(YT, ST)])
        return YF
