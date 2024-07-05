import numpy as np
import pandas as pd
import scipy
from holisticai.bias.mitigation.postprocessing.lp_debiaser.multiclass_balancer import algorithm_utils as tools


class MulticlassBalancerAlgorithm:
    def __init__(self, constraint, objective):
        """
        Initializes an instance of a PredictionBalancer.
        """
        self.constraint = constraint
        self.objective = objective

    def fit(self, y_true, y_pred, p_attr):
        """
        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or str
            The true labels, either as an array or as a string \
            specifying a column in data.

        y_pred : array-like of shape (n_samples,) or str
            The predicted labels, either as an array or as a string \
            specifying a column in data.

        p_attr : array-like of shape (n_samples,) or str
            The protected attribute, either as an array, or as a string \
            specifying the column in data.
        """

        assert y_pred is not None, "y_pred must be passed"

        # Getting the group info
        p_y = tools.p_vec(y_true)
        p_a = tools.p_vec(p_attr)
        self.n_classes = p_y.shape[0]

        # Getting some basic info for each group
        self.groups = np.unique(p_attr)
        self.n_groups = len(self.groups)
        group_ids = [np.where(p_attr == g)[0] for g in self.groups]

        # Getting the group-specific P(Y), P(Y- | Y), and constraint matrices
        p_vecs = np.array([tools.p_vec(y_true[ids]) for ids in group_ids])
        p_vecs = p_a.reshape(-1, 1) * p_vecs
        cp_mats = np.array([tools.cp_mat(y_true[ids], y_pred[ids], self.n_classes) for ids in group_ids])
        cp_mats_t = np.zeros((self.n_classes, self.n_classes, self.n_groups))

        for a in range(self.n_groups):
            cp_mats_t[:, :, a] = cp_mats[a].transpose()

        self.constraint.n_groups = self.n_groups
        self.objective.n_classes = self.n_classes
        self.objective.n_groups = self.n_groups

        con, con_bounds = self.constraint.get_conditions(p_vecs, p_a, cp_mats)
        self.obj = self.objective(cp_mats)

        # Running the optimization
        opt = scipy.optimize.linprog(c=self.obj, bounds=[0, 1], A_eq=con, b_eq=con_bounds, method="highs")

        # Getting the Y~ matrices
        m = tools.pars_to_cpmat(opt, n_groups=self.n_groups, n_classes=self.n_classes)

        # Getting the new cp matrices
        self.new_cp_mats = np.array([np.dot(cp_mats[i], m[i]) for i in range(self.n_groups)])

    def predict(self, y_pred, p_attr, seed=2021):
        """Generates bias-adjusted predictions on new data.

        Parameters
        ----------
        y_pred : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.

        p_attr : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.

        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        assert y_pred is not None, "y_pred must be passed"

        pd.options.mode.chained_assignment = None
        y_tilde = y_pred.copy()
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, len(self.groups))
        for g in self.groups:
            for c in range(self.n_classes):
                y_ids = np.where((p_attr == g) & (y_pred == c))[0]
                np.random.seed(seeds[g])
                y_tilde[y_ids] = np.random.choice(a=self.n_classes, p=self.new_cp_mats[g][c], size=len(y_ids))
        return y_tilde
