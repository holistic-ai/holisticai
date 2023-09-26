class GlobalFeatureImportance:
    def partial_dependence_plot(self, first=0, last=None, **plot_kargs):

        import matplotlib.pyplot as plt
        from sklearn.inspection import PartialDependenceDisplay

        top_k = None
        if last == None:
            last = first + 6

        importances = self.get_topk(top_k=top_k)
        fimp = importances["feature_importance"]
        features = list(fimp["Variable"])[first:last]
        title = "Partial dependence plot"
        percentiles = (
            (0, 1) if self.model_type == "binary_classification" else (0.05, 0.95)
        )

        common_params = {
            "subsample": 50,
            "n_jobs": 2,
            "grid_resolution": 20,
            "random_state": 0,
            "kind": "average",
            "percentiles": percentiles,
        }

        common_params.update(plot_kargs)

        plt.rcParams["figure.constrained_layout.use"] = True

        pdp = PartialDependenceDisplay.from_estimator(
            self.model,
            self.x,
            features,
            **common_params,
        )
        pdp.figure_.suptitle(title)
        plt.show()


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
