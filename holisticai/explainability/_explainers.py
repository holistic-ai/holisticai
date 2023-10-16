import warnings

import matplotlib.pyplot as plt
import pandas as pd

from holisticai.explainability.metrics.utils import check_feature_importance
from holisticai.explainability.plots import bar, contrast_matrix, lolipop

from .metrics.extractors.local_feature_importance import (
    compute_local_feature_importance,
)
from .metrics.extractors.permutation_feature_importance import (
    compute_permutation_feature_importance,
)
from .metrics.extractors.surrogate_feature_importance import (
    compute_surrogate_feature_importance,
)
from .metrics.global_importance._contrast_metrics import important_constrast_matrix

warnings.filterwarnings("ignore")


class Explainer:
    def __init__(self, based_on, strategy_type, model_type, x, y, **kargs):
        """
        The Explainer class is designed to compute various feature importance strategies and explainability metrics based on different types of models.

        Parameters
        ----------
        based_on: str
            Indicates the basis for explainability computation. It can be "feature_importance".

        strategy_type: str
            The strategy to be used for explainability computation, such as "permutation", "surrogate", "local", "lime", or "shap".

        model_type: str
            The type of model used for explainability.

        x: matrix-like
            The feature dataset.

        y: array-like
            The target dataset.

        **kargs: Additional arguments specific to each explainability strategy.

        Note:
        - This class supports various explainability strategies, including permutation, surrogate, local, Lime, and Shap, depending on the provided options.
        - The primary goal is to provide a convenient interface for computing and storing feature importance explanations and explainability metrics for machine learning models.

        Example usage:

        ```
        explainer = Explainer(based_on="feature_importance", strategy_type="permutation", model_type="regression", x=X, y=y, model=my_model)
        explainer.metrics()
        ```
        """

        self.model_type = model_type
        model = kargs.get("model", None)

        if based_on == "feature_importance":
            x, y = check_feature_importance(x, y)

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

            elif strategy_type == "local":
                self.explainer_handler = compute_local_feature_importance(
                    model_type, x, y, **kargs
                )
                self._strategy_type = "local"

            elif strategy_type == "lime":
                self.check_installed_package("lime")

                from holisticai.explainability.metrics.utils import LimeTabularHandler

                local_explainer_handler = LimeTabularHandler(
                    model.predict,
                    x.values,
                    feature_names=x.columns.tolist(),
                    discretize_continuous=True,
                    mode="regression",
                )
                self.explainer_handler = compute_local_feature_importance(
                    model_type, x, y, local_explainer_handler=local_explainer_handler
                )
                self._strategy_type = "local"

            elif strategy_type == "shap":
                self.check_installed_package("shap")
                import shap

                from holisticai.explainability.metrics.utils import ShapTabularHandler

                X100 = shap.utils.sample(x, 100)
                local_explainer_handler = ShapTabularHandler(model.predict, X100)
                self.explainer_handler = compute_local_feature_importance(
                    model_type, x, y, local_explainer_handler=local_explainer_handler
                )
                self._strategy_type = "local"

            else:
                raise NotImplementedError

    def check_installed_package(self, backend):
        import importlib

        allowed_packages = ["shap", "lime"]
        backend_package = importlib.util.find_spec(backend)
        if (backend and allowed_packages) and (backend_package is None):
            raise (
                "Package {backend} must be installed. Please install with: pip install {backend}"
            )

    def __getitem__(self, key):
        return self.metric_values.loc[key]["Value"]

    def metrics(self, top_k=None, detailed=False):
        """
        top_k: int
            Number of features to select
        """
        params = self.explainer_handler.get_topk(top_k)
        self.metric_values = self.explainer_handler.metrics(**params, detailed=detailed)
        return self.metric_values

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
        Parameters
        ----------

        backend: str
            - sklearn: check sklearn.tree.plot_tree input parameters.
            - pydotplus: check export_graphviz input parameters.
            - dtreeviz: check dtreeviz.model.view input parameters.
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
        self.explainer_handler.partial_dependence_plot(
            first=first, last=last, **plot_kargs
        )

    def show_importance_stability(self):
        importances = self.explainer_handler.get_topk(top_k=None)
        cfimp = importances["conditional_feature_importance"]
        fimp = importances["feature_importance"]
        self.explainer_handler.show_importance_stability(fimp, cfimp)

    def show_data_stability_boundaries(
        self, n_rows=2, n_cols=4, top_n=None, figsize=None
    ):
        self.explainer_handler.show_data_stability_boundaries(
            n_rows, n_cols, top_n=top_n, figsize=figsize
        )

    def feature_importance_table(self, sorted_by="Global", top_n=10):
        feature_importance = self.explainer_handler.get_topk(None)
        dfs = []
        df = (
            feature_importance["feature_importance"][["Variable", "Importance"]]
            .reset_index(drop=True)
            .set_index("Variable")
        )
        df.columns = ["Global Importance"]
        dfs.append(df)

        if "conditional_feature_importance" in feature_importance:
            for name, cfi in feature_importance[
                "conditional_feature_importance"
            ].items():
                cdf = (
                    cfi[["Variable", "Importance"]]
                    .reset_index(drop=True)
                    .set_index("Variable")
                )
                cdf.columns = [f"{name} Importance"]
                dfs.append(cdf)

        dfs = pd.concat(dfs, axis=1).sort_values(
            f"{sorted_by} Importance", ascending=False
        )
        subset = [col for col in dfs.columns if "Importance" in col]
        vmax = dfs[subset].max().max()
        dfs = dfs.iloc[:top_n].style.bar(
            subset=subset, color="lightgreen", vmin=0.0, vmax=vmax
        )
        return dfs

    def show_features_stability_boundaries(
        self, n_cols=None, n_rows=None, figsize=None
    ):
        self.explainer_handler.show_features_stability_boundaries(
            n_cols=n_cols, n_rows=n_rows, figsize=figsize
        )
