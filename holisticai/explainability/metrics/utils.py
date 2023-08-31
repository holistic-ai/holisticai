import numpy as np
import pandas as pd
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay


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
        index_groups = {f"[label={value}]": y[y == value].index for value in y.unique()}
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


def lime_creator(
    scorer,
    X,
    index_groups=None,
    num_features=None,
    num_samples=None,
    mode="classification",
):
    """
    Parameters
    ----------
    scorer: sklearn-like scorer
        scorer function
    X: np.array
        input data
    index_groups: dict
        dictionary with groups
    num_features: int
        number of features to select
    num_samples: int
        number of samples to select
    mode: str
        classification or regression
    """
    # load and do assignment
    if num_features is None:
        num_features = np.min([X.shape[1], 50])

    if num_samples is None:
        num_samples = np.min([X.shape[0], 50])

    per_group_sample = int(np.ceil(num_samples / len(index_groups)))
    ids_groups = {
        str(label): np.random.choice(X.index[index], size=per_group_sample).tolist()
        for label, index in index_groups.items()
    }

    # calculate lime for several samples
    explainer = lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        discretize_continuous=True,
        mode=mode,
    )

    df = []
    for label, indexes in ids_groups.items():
        for i in indexes:
            exp = explainer.explain_instance(
                X.loc[i], scorer, num_features=X.shape[1], num_samples=100
            )
            exp_values = list(exp.local_exp.values())[0]

            df_i = pd.DataFrame(exp_values, columns=["Feature Id", "Feature Weight"])
            df_i["Importance"] = df_i["Feature Weight"].abs()
            df_i["Importance"] = df_i["Importance"] / df_i["Importance"].max()
            df_i["Sample Id"] = i
            df_i["Feature Label"] = X.columns[df_i["Feature Id"].tolist()]
            df_i["Feature Rank"] = range(1, df_i.shape[0] + 1)
            df_i["Sample Group"] = label
            df.append(df_i)

    df = pd.concat(df, axis=0, ignore_index=True)

    return df


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


def four_fifths_list(feature_importance, cutoff=None):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    cutoff: float
        threshold for feature importance
    """
    if cutoff is None:
        cutoff = 0.80

    importance = feature_importance["Importance"]
    feature_names = feature_importance["Variable"]

    feature_weight = importance / sum(importance)

    # entropy or divergence
    return feature_names.loc[(feature_weight.cumsum() < cutoff).values]


def get_top_k(df_feature_importance, top_k):
    feat_id = four_fifths_list(df_feature_importance, cutoff=top_k)
    df_feature_importance = df_feature_importance.loc[
        df_feature_importance["Variable"].isin(list(feat_id))
    ]

    return df_feature_importance


def get_top_k_lime(df_feature_importance, top_k):
    feat_imp = df_feature_importance["Importance"]
    feat_names = df_feature_importance["Feature Label"]

    feat_id = four_fifths_list_lime(feat_imp, feat_names, cutoff=top_k)
    df_feature_importance = df_feature_importance.loc[
        df_feature_importance["Feature Label"].isin(list(feat_id))
    ]

    return df_feature_importance


def partial_dependence_creator(model, grid_resolution, x, feature_ids, target=None):
    """
    Parameters
    ----------
    model: sklearn-like object
        sklearn-like object with predict method
    grid_resolution: int
        number of points to compute partial dependence
    X: np.array
        input data
    feature_ids: list
        list of feature ids
    target: int
        target class
    """
    # to do
    # -> explicit implementation of plot_partial_dependence_plot from sklearn without showing the chart
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["interactive"] == False

    feature_names = list(x.columns)
    method = "brute"
    percentiles = (0.05, 0.95)
    response_method = "auto"

    kargs = {
        "estimator": model,
        "X": x,
        "features": feature_ids,
        "feature_names": feature_names,
        "percentiles": percentiles,
        "response_method": response_method,
        "method": method,
        "grid_resolution": grid_resolution,
    }

    if not target == None:
        kargs.update({"target": target})

    g = PartialDependenceDisplay.from_estimator(**kargs)
    plt.close()

    pd_results = {}
    for (i, f) in enumerate(feature_ids):
        pd_results[feature_names[f]] = pd.DataFrame(
            {
                "score": g.pd_results[i]["average"][0],
                "values": g.pd_results[i]["values"][0],
            }
        )

    return pd_results


def importance_spread(feature_importance, divergence=False):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    divergence: bool
        if True calculate divergence instead of ratio
    """
    if len(feature_importance) == 0:
        return 0 if divergence else 1

    importance = feature_importance
    from scipy.stats import entropy

    feature_weight = importance / sum(importance)
    feature_equal_weight = np.array([1.0 / len(importance)] * len(importance))

    # entropy or divergence
    if divergence is False:
        return entropy(feature_weight) / entropy(feature_equal_weight)  # ratio
    else:
        return entropy(feature_weight, feature_equal_weight)  # divergence


def explanation_contrast(feature_importance, cond_feature_importance, order=True):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    cond_feature_importance: np.array
        array with raw feature importance
    order: bool
        if True calculate order contrast
    """
    # todo - implement per weight
    # in case we are per measuring order
    if order:
        conditionals = list(cond_feature_importance.keys())
        results = []
        for c in conditionals:
            matching = (
                feature_importance["Variable"].index
                == cond_feature_importance[c]["Variable"].index
            )
            results += (c, matching.mean())

    return results


def importance_range_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_feature_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    m_range = []
    for top_k in range(1, len(feature_importance_indexes) + 1):
        ggg = set(feature_importance_indexes[:top_k])
        vvv = set(conditional_feature_importance_indexes[:top_k])
        u = len(set(ggg).intersection(vvv)) / top_k
        m_range.append(u)
    m_range = np.array(m_range)

    return m_range.mean()


def importance_order_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_features_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    m_order = np.array(feature_importance_indexes) == np.array(
        conditional_features_importance_indexes
    )
    m_order = np.cumsum(m_order) / np.arange(1, len(m_order) + 1)

    return m_order.mean()
