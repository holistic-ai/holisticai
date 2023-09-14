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
            f"[label={int(value)}]": y[y == value].index for value in y.unique()
        }
        return index_groups

    elif model_type == "regression":
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(y.quantile(labels_values)).squeeze()
        index_groups = {
            f"[{c}]": y[(y.values > v[i]) & (y.values < v[i + 1])].index
            for (i, c) in enumerate(labels)
        }
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
    pass


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
