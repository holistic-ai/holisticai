import numpy as np
import pandas as pd

from ..utils.explainer_utils import four_fifths_list


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
    return feature_importance_names.loc[(feature_weight.cumsum() < cutoff).values]


def quantil_classify(q1, q2, q3, labels, x):
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

        groups = y.apply(lambda x: quantil_classify(v[1], v[2], v[3], labels, x))

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
