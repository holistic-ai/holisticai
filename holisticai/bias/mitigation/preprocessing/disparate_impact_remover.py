import numpy as np

from holisticai.bias.mitigation.commons.disparate_impact_remover._numerical_repairer import (
    NumericalRepairer,
)
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from holisticai.utils.transformers.bias import SensitiveGroups


class DisparateImpactRemover(BMPre):
    """
    Disparate impact remover edits feature values to increase group fairness
    while preserving rank-ordering within groups.

    References
    ----------
        [1] Feldman, Michael, et al. "Certifying and removing disparate impact."
        proceedings of the 21th ACM SIGKDD international conference on knowledge
        discovery and data mining. 2015.
    """

    def __init__(self, repair_level=1.0):
        """
        Disparate Impact Remover Preprocessing Bias Mitigator

        Description
        -----------
        Initialize Mitigator class

        Parameters
        ----------
        repair_level : float
                Repair amount 0.0 (min) -> 1.0 (max)
        """
        self._assert_parameters(repair_level)
        self.repair_level = repair_level
        self.sensgroup = SensitiveGroups()

    def _assert_parameters(self, repair_level):
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")

    def repair_data(self, X, group_a, group_b):
        sensitive_features = np.c_[group_a, group_b]
        p_attr = self.sensgroup.fit_transform(
            sensitive_features, convert_numeric=True
        ).to_numpy()
        data = np.c_[p_attr, X].tolist()
        dir = NumericalRepairer(feature_to_repair=0, repair_level=1.0, kdd=False)
        new_data_matrix_np = dir.repair(data)
        return np.array([np.array(row[1:]) for row in new_data_matrix_np])

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Transform data

        Description
        -----------
        Transform data to a fair representation

        Parameters
        ----------
        X : ndarray
            Input data

        group_a : ndarray
            mask vector

        group_b : ndarray
            mask vector

        Return
        ------
            Self
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        return self.repair_data(X, group_a, group_b)

    def fit(self):
        return self

    def fit_transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        return self.fit().transform(X, group_a, group_b)
