import warnings
import matplotlib.pyplot as plt
import pandas as pd

from holisticai.explainability.plots import (
    bar,
    contrast_matrix,
    lolipop,
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
        self.model_type = model_type
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

    def tree_visualization(self, backend, **kargs):
        """
        Aditional parameters:
        for backend:
            sklearn: check sklearn.tree.plot_tree input parameters.
            pydotplus: check export_graphviz input parameters.
            dtreeviz: check dtreeviz.model.view input parameters.
        """
        return self.explainer_handler.tree_visualization(backend, **kargs)

    def contrast_visualization(self, show_connections=False):
        importances = self.explainer_handler.get_topk(top_k=None)
        cfimp = importances["conditional_feature_importance"]
        fimp = importances["feature_importance"]
        keys = list(cfimp.keys())
        xticks, matrix = important_constrast_matrix(
            cfimp, fimp, keys, show_connections=show_connections
        )
        contrast_matrix(xticks, matrix)

    
    def partial_dependence_plot(self, first=0, last=None, **plot_kargs):
        self.explainer_handler.partial_dependence_plot(first=first, last=last, **plot_kargs)

    def show_importance_stability(self):
        importances = self.explainer_handler.get_topk(top_k=None)
        cfimp = importances["conditional_feature_importance"]
        fimp = importances["feature_importance"]
        self.explainer_handler.show_importance_stability(fimp, cfimp)

    def show_data_stability_boundaries(self, top_n=None, figsize=None):
        self.explainer_handler.show_data_stability_boundaries(top_n=top_n, figsize=figsize)
    
    def feature_importance_table(self, sorted_by='Global', top_n=10):
        feature_importance = self.explainer_handler.get_topk(None)
        dfs = []
        df = feature_importance['feature_importance'][['Variable','Importance']].reset_index(drop=True).set_index('Variable')
        df.columns = ['Global Importance']
        dfs.append(df)
        
        if 'conditional_feature_importance' in feature_importance:
            for name,cfi in feature_importance['conditional_feature_importance'].items():
                cdf = cfi[['Variable','Importance']].reset_index(drop=True).set_index('Variable')
                cdf.columns = [f'{name} Importance']
                dfs.append(cdf)
            
        dfs = pd.concat(dfs,axis=1).sort_values(f'{sorted_by} Importance', ascending=False)
        subset = [col for col in dfs.columns if 'Importance' in col]
        vmax = dfs[subset].max().max()
        dfs = dfs.iloc[:top_n].style.bar(subset=subset, color='lightgreen', vmin=0.0, vmax=vmax)
        return dfs
    
    
    def show_features_stability_boundaries(self, figsize=None):
        self.explainer_handler.show_features_stability_boundaries(figsize=figsize)