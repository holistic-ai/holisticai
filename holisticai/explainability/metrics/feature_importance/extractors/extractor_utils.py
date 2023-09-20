import numpy as np
import pandas as pd

from ..utils import four_fifths_list


def four_fifths_list_lime(feature_importance, feature_importance_names, cutoff=None):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    feature_importance_names: list
        list with names
    cutoff: float
        threshold for feature importance
    """
    if cutoff is None:
        cutoff = 0.80

    feature_weight = feature_importance / sum(feature_importance)

    # entropy or divergence
    return feature_importance_names.loc[(feature_weight.cumsum() < cutoff).values]


def classify_element(q1, q2, q3, labels, x):
    if x <= q1:
        return labels[0]
    elif q1 < x <= q2:
        return labels[1]
    elif q2 < x <= q3:
        return labels[2]
    else:
        return labels[3]


def get_index_groups(model_type, y):
    """
    Parameters
    ----------
    model_type: str
        type of model
    y: np.array
        target array
    """
    if model_type == "binary_classification":
        index_groups = {
            f"[label={int(value)}]": list(y[y == value].index) for value in y.unique()
        }
        return index_groups

    elif model_type == "regression":
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(y.quantile(labels_values)).squeeze()

        groups = y.apply(lambda x: classify_element(v[1], v[2], v[3], labels, x))

        df = pd.concat([y, groups], axis=1)
        df.columns = ["y", "group"]
        index_groups = {f"[{k}]": list(v.index) for k, v in df.groupby("group")["y"]}

        return index_groups
    else:
        raise NotImplementedError


def get_top_k(df_feature_importance, top_k):
    feat_id = four_fifths_list(df_feature_importance, cutoff=top_k)
    df_feature_importance = df_feature_importance.loc[
        df_feature_importance["Variable"].isin(list(feat_id))
    ]
    return df_feature_importance


def get_top_k_lime(df_feature_importance, top_k):

    mean_importance = (
        df_feature_importance.groupby("Feature Label")["Importance"]
        .mean()
        .reset_index("Feature Label")
    )
    feat_imp = mean_importance["Importance"]
    feat_names = mean_importance["Feature Label"]
    feat_id = four_fifths_list_lime(feat_imp, feat_names, cutoff=top_k)

    df_feature_importance = df_feature_importance.loc[
        df_feature_importance["Feature Label"].isin(list(feat_id))
    ]
    return df_feature_importance


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
        percentiles =  (0,1) if self.model_type=='binary_classification' else (0.05, 0.95)

        common_params = {
            "subsample": 50,
            "n_jobs": 2,
            "grid_resolution": 20,
            "random_state": 0,
            "kind":"average",
            "percentiles":percentiles
        }
        
        common_params.update(plot_kargs)
        
        plt.rcParams['figure.constrained_layout.use'] = True
        
        pdp = PartialDependenceDisplay.from_estimator(
            self.model, self.x, features, **common_params,
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
