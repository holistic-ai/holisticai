from ..utils import alpha_feature_importance


class GlobalFeatureImportance:
    def get_alpha_feature_importance(self, alpha=0.8):

        feat_imp = alpha_feat_imp = self.feature_importance.set_index(
            "Variable"
        ).sort_values("Importance", ascending=False)

        if alpha is not None:
            alpha_feat_imp = alpha_feature_importance(feat_imp, alpha)

        alpha_cond_feat_imp = None
        cond_feat_imp = None
        if hasattr(self, "conditional_feature_importance") and (
            self.conditional_feature_importance is not None
        ):
            cond_feat_imp = {
                label: value.set_index("Variable").sort_values(
                    "Importance", ascending=False
                )
                for label, value in self.conditional_feature_importance.items()
            }

            alpha_cond_feat_imp = {
                label: alpha_feature_importance(value, alpha)
                for label, value in self.conditional_feature_importance.items()
            }

        return (feat_imp, cond_feat_imp), (alpha_feat_imp, alpha_cond_feat_imp)

    def partial_dependence_plot(self, first=0, last=None, alpha=0.8, **plot_kargs):

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.inspection import PartialDependenceDisplay

        sns.set_style("whitegrid")

        if last == None:
            last = first + 6

        importances = self.get_alpha_feature_importance(alpha)
        (feat_imp, _), (alpha_feat_imp, alpha_cond_feat_imp) = importances

        from ..global_importance import ExplainabilityEase

        expe = ExplainabilityEase(
            model_type=self.model_type, model=self.model, x=self.x
        )
        _, score_data = expe(alpha_feat_imp, return_score_data=True)
        features = list(alpha_feat_imp.index)[first:last]
        level = [score_data.loc[f]["scores"] for f in features]
        title = "Partial dependence plot"
        percentiles = (
            (0, 1) if self.model_type == "binary_classification" else (0.05, 0.95)
        )

        common_params = {
            "subsample": 50,
            "n_jobs": 2,
            "grid_resolution": 50,
            "random_state": 0,
            "kind": "average",
            "percentiles": percentiles,
            "line_kw": {"color": "mediumslateblue", "label": "Average"},
        }

        common_params.update(plot_kargs)

        plt.rcParams["figure.constrained_layout.use"] = True

        pdp = PartialDependenceDisplay.from_estimator(
            self.model,
            self.x,
            features,
            **common_params,
        )

        for lv, ax in zip(level, pdp.axes_[0]):
            ax.legend([lv])

        acc_feat_imp = np.sum(alpha_feat_imp["Importance"])
        num_features = len(alpha_feat_imp)
        print(
            f"The accumulated feature importance for {num_features} features is equal {acc_feat_imp:.4f} < {alpha}."
        )
        pdp.figure_.suptitle(title)

        return pdp


class LocalFeatureImportance:
    pass


class BaseFeatureImportance:
    def __init__(
        self,
        model_type,
        model,
        x,
        y,
        importance_weights,
        conditional_importance_weights,
    ):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.importance_weights = importance_weights
        self.conditional_importance_weights = conditional_importance_weights

    def custom_metrics(self):
        pass
