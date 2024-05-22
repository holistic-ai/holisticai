import numpy as np
import scipy as sp
from sklearn.metrics import roc_curve

from ..binary_balancer import algorithm_utils as tools


class BinaryBalancerAlgorithm:
    def __init__(self, constraint, objective, threshold_objective="j", binom=False):
        self.constraint = constraint
        self.thr_obj = threshold_objective
        self.objective = objective
        self.binom = binom

    def fit(self, y_true, p_attr, y_pred=None, y_proba=None):

        assert not (
            (y_pred is None) and (y_proba is None)
        ), f"y_pred or y_proba must be passed"

        # Getting the group info
        self.groups = np.unique(p_attr)
        group_ids = [np.where(p_attr == g)[0] for g in self.groups]

        # Optionally thresholding probabilities to get class predictions
        if y_proba is not None:
            probs = y_proba[:, 1]
            self.rocs = [roc_curve(y_true[ids], probs[ids]) for ids in group_ids]
            self.__roc_stats = [
                tools.loss_from_roc(y_true[ids], probs[ids], self.rocs[i])
                for i, ids in enumerate(group_ids)
            ]
            if self.thr_obj == "j":
                cut_ids = [np.argmax(rs["js"]) for rs in self.__roc_stats]
                self.cuts = [self.rocs[i][2][id] for i, id in enumerate(cut_ids)]
                for g, cut in enumerate(self.cuts):
                    probs[group_ids[g]] = tools.threshold(probs[group_ids[g]], cut)
                y_pred = probs.astype(np.uint8)

        # Calcuating the groupwise classification rates
        rates_by_group = [tools.CLFRates(y_true[i], y_pred[i]) for i in group_ids]
        self.group_rates = dict(zip(self.groups, rates_by_group))

        # And then the overall rates
        self.overall_rates = tools.CLFRates(y_true, y_pred)

        # Getting the coefficients for the objective
        p = [len(cols) / len(y_true) for cols in group_ids]

        dr = [(g.nr * p[i], g.pr * p[i]) for i, g in enumerate(rates_by_group)]

        obj_coefs = self.objective(dr, self.overall_rates)

        con, con_bounds = self.constraint.get_conditions(self.group_rates, self.groups)

        # Running the optimization
        self.opt = sp.optimize.linprog(
            c=obj_coefs, bounds=[(0, 1)], A_eq=con, b_eq=con_bounds, method="highs"
        )

        self.pya = self.opt.x.reshape(len(self.groups), 2)

        # Setting the adjusted predictions
        self.y_adj = tools.pred_from_pya(
            y_pred=y_pred, p_attr=p_attr, pya=self.pya, binom=self.binom
        )

    def predict(self, p_attr, y_pred=None, y_proba=None, binom=False):
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if y_proba is not None:
            probs = y_proba[:, 1]
            group_ids = [np.where(p_attr == g)[0] for g in self.groups]
            for g, cut in enumerate(self.cuts):
                y_pred[group_ids[g]] = tools.threshold(probs[group_ids[g]], cut)

        # Returning the adjusted predictions
        adj = tools.pred_from_pya(y_pred, p_attr, self.pya, binom)
        return adj
