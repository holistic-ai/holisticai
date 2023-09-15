import warnings

import matplotlib.pyplot as plt

from holisticai.explainability.plots import (
    bar,
    contrast_matrix,
    lolipop,
    partial_dependence_plot,
)

from .metrics.feature_importance.extractors.lime_feature_importance import (
    compute_lime_feature_importance,
)
from .metrics.feature_importance.extractors.permutation_feature_importance import (
    compute_permutation_feature_importance,
)
from .metrics.feature_importance.extractors.surrogate_feature_importance import (
    compute_surrogate_feature_importance,
)
from .metrics.feature_importance.global_importance._contrast_metrics import (
    important_constrast_matrix,
)

warnings.filterwarnings("ignore")


class Explainer:
    def __init__(self, based_on, strategy_type, model_type, model, x, y):
        if based_on == "feature_importance":

            if strategy_type == "permutation":
                self.explainer_handler = compute_permutation_feature_importance(
                    model_type, model, x, y
                )
                self._strategy_type = "global"

            elif strategy_type == "surrogate":
                self.explainer_handler = compute_surrogate_feature_importance(
                    model_type, model, x, y
                )
                self._strategy_type = "global"

            elif strategy_type == "lime":
                self.explainer_handler = compute_lime_feature_importance(
                    model_type, model, x, y
                )
                self._strategy_type = "local"

            else:
                raise NotImplementedError

    def metrics(self, top_k=None, detailed=False):
        """
        top_k: int
            Number of features to select
        """
        params = self.explainer_handler.get_topk(top_k)
        return self.explainer_handler.metrics(**params, detailed=detailed)

    def bar_plot(self, max_display=None, title=None, top_k=None, figsize=(7, 5)):
        """
        Parameters
        ----------
        max_display: int
            Maximum number of features to display
        title: str
            Title of the plot
        top_k: int
            Number of features to select
        figsize: tuple
            Size of the plot
        """
        params = self.explainer_handler.get_topk(top_k)
        feat_imp = params["feature_importance"]
        bar(
            feat_imp,
            max_display=max_display,
            title=title,
            figsize=figsize,
            _type=self._strategy_type,
        )

    def lolipop_plot(self, max_display=None, title=None, top_k=None, figsize=(7, 5)):
        """
        Parameters
        ----------
        max_display: int
            Maximum number of features to display
        title: str
            Title of the plot
        top_k: int
            Number of features to select
        figsize: tuple
            Size of the plot
        """
        params = self.explainer_handler.get_topk(top_k)
        feat_imp = params["feature_importance"]
        lolipop(
            feat_imp,
            max_display=max_display,
            title=title,
            figsize=figsize,
            _type=self._strategy_type,
        )

    def tree_visualization(self, backend):
        return self.explainer_handler.tree_visualization(backend)

    def contrast_visualization(self, show_connections=False):
        importances = self.explainer_handler.get_topk(top_k=None)
        cfimp = importances["conditional_feature_importance"]
        fimp = importances["feature_importance"]
        keys = list(cfimp.keys())
        xticks, matrix = important_constrast_matrix(
            cfimp, fimp, keys, show_connections=show_connections
        )
        contrast_matrix(xticks, matrix)

    def partial_dependence_plot(self, grid_resolution=20, top_k=0.8, ax=None):
        plt.rcParams["figure.constrained_layout.use"] = True

        importances = self.explainer_handler.get_topk(top_k=top_k)
        fimp = importances["feature_importance"]

        features = list(fimp["Variable"])

        model = self.explainer_handler.model
        x = self.explainer_handler.x

        title = "Partial dependence plot"
        return partial_dependence_plot(
            x, features, title, model, grid_resolution=grid_resolution, ax=ax
        )

    def show_importance_stability(self):
        importances = self.explainer_handler.get_topk(top_k=None)
        cfimp = importances["conditional_feature_importance"]
        fimp = importances["feature_importance"]
        return self.explainer_handler.show_importance_stability(fimp, cfimp)
