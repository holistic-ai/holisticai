import os
import sys

sys.path.append(os.getcwd())
import numpy as np

from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithm_utils import f_lambda


class PluginEstimationAndCalibrationAlgorithm:
    def __init__(self, L=25, beta=0.1):
        self.M = 1
        self.L = np.floor(L / 2).astype(np.int32)
        self.beta = beta
        self.sens_groups = SensitiveGroups()
        self.eps = np.finfo(float).eps

    def fit(self, y_pred: np.ndarray, sensitive_features: np.ndarray):
        SL = self.sens_groups.fit_transform(sensitive_features, convert_numeric=True)
        YL = y_pred * 2 - 1
        self.p = [np.mean(SL == 0), np.mean(SL == 1)]
        self.lambda_ = f_lambda(YL, SL, self.M, self.L, self.beta)

    def transform(self, y_pred: np.ndarray, sensitive_features: np.ndarray):
        ST = self.sens_groups.transform(sensitive_features, convert_numeric=True)
        YT = y_pred * 2 - 1
        i = np.arange(-self.L, self.L + 1, 1)
        YTd = np.zeros(len(YT))
        for k in range(len(YT)):
            v = (
                self.p[ST[k]] * (YT[k] - i * self.M / self.L) ** 2
                + (1 - 2 * ST[k]) * self.lambda_
            )
            j = np.argmin(v)
            YTd[k] = i[j] * self.M / self.L
        YTd = (YTd + 1) / 2
        return YTd
