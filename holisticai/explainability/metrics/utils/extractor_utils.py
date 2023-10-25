import numpy as np
import pandas as pd

from ..utils.explainer_utils import alpha_importance_list_lime


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


def get_alpha_lime(df_feature_importance, alpha):

    mean_importance = (
        df_feature_importance.groupby("Feature Label")["Importance"]
        .mean()
        .reset_index("Feature Label")
    )
    feat_imp = mean_importance["Importance"]
    feat_names = mean_importance["Feature Label"]
    feat_id = alpha_importance_list_lime(feat_imp, feat_names, alpha=alpha)

    df_feature_importance = df_feature_importance.loc[
        df_feature_importance["Feature Label"].isin(list(feat_id))
    ]
    return df_feature_importance
