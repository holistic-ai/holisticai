import numpy as np
from holisticai.bias.mitigation.commons.disparate_impact_remover._numerical_repairer import (
    NumericalRepairer,
)
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from holisticai.utils.transformers.bias import SensitiveGroups


class DisparateImpactRemover(BMPre):
    """
    Disparate impact remover [1]_ edits feature values to increase group fairness
    while preserving rank-ordering within groups.

    Parameters
    ----------
    repair_level : float, optional
        The amount of repair to be applied. It should be between 0.0 and 1.0. Default is 1.

    References
    ----------
    .. [1] Feldman, Michael, et al. "Certifying and removing disparate impact."
        proceedings of the 21th ACM SIGKDD international conference on knowledge
        discovery and data mining. 2015.
    """

    def __init__(self, repair_level=1.0):
        self._assert_parameters(repair_level)
        self.repair_level = repair_level
        self._sensgroups = SensitiveGroups()

    def _assert_parameters(self, repair_level):
        """
        Assert parameters

        Parameters
        ----------
        repair_level : float
            The amount of repair to be applied

        Raises
        ------
        ValueError
            If repair_level is not between 0.0 and 1.0
        """
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")

    def repair_data(self, X, group_a, group_b):
        """
        Repair data to a fair representation

        Parameters
        ----------
        X : array-like
            Input data
        group_a : array-like
            mask vector
        group_b : array-like
            mask vector

        Returns
        -------
        array-like
            Repaired input data matrix
        """
        # Combine the two group masks into a single matrix
        sensitive_features = np.c_[group_a, group_b]

        # Convert the sensitive feature matrix to a numeric representation
        p_attr = self._sensgroups.fit_transform(sensitive_features, convert_numeric=True).to_numpy()

        # Combine the sensitive feature matrix with the input data matrix
        data = np.c_[p_attr, X].tolist()

        # Apply numerical repair to the first column of the combined matrix
        repairer = NumericalRepairer(feature_to_repair=0, repair_level=self.repair_level, kdd=False)
        new_data_matrix_np = repairer.repair(data)

        # Return only the repaired input data matrix (without the sensitive feature column)
        return np.array([np.array(row[1:]) for row in new_data_matrix_np])

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Transform data to a fair representation

        Parameters
        ----------
        X : array-like
            Input data

        group_a : array-like
            mask vector

        group_b : array-like
            mask vector

        Returns
        -------
            Self
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        x = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        return self.repair_data(x, group_a, group_b)

    def fit(self):
        """
        Fit the model

        Returns
        -------
            Self
        """
        return self

    def fit_transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Fit the model and transform data

        Parameters
        ----------
        X : array-like
            Input data
        group_a : array-like
            mask vector
        group_b : array-like
            mask vector

        Returns
        -------
        array-like
            Repaired input data matrix
        """
        return self.fit().transform(X, group_a, group_b)
