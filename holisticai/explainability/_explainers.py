import warnings

from holisticai.explainability.plots import bar, lolipop

from .metrics.feature_importance.extractors.lime_feature_importance import (
    compute_lime_feature_importance,
)
from .metrics.feature_importance.extractors.permutation_feature_importance import (
    compute_permutation_feature_importance,
)
from .metrics.feature_importance.extractors.surrogate_feature_importance import (
    compute_surrogate_feature_importance,
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

    def metrics(self, top_k=None):
        """
        top_k: int
            Number of features to select
        """
        params = self.explainer_handler.get_topk(top_k)
        return self.explainer_handler.metrics(**params)

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

    def visualization(self, visualization_type):
        return self.explainer_handler.visualization(visualization_type)
