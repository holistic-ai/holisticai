import numpy as np
import pandas as pd
from lime import lime_tabular
from tqdm import tqdm

from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)

from ..local_importance._local_metrics import (
    dataset_spread_stability,
    features_spread_stability,
)
from .extractor_utils import (
    BaseFeatureImportance,
    LocalFeatureImportance,
    get_index_groups,
    get_top_k_lime,
)


def compute_lime_feature_importance(model_type, model, x, y):
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    if not isinstance(y, pd.Series):
        y = _array_like_to_series(y)

    if model_type == "binary_classification":
        lime_mode = "classification"
        scorer = model.predict_proba

    elif model_type == "regression":
        lime_mode = "regression"
        scorer = model.predict

    index_groups = get_index_groups(model_type, y)
    features_importance = lime_creator(
        scorer=scorer, X=x, index_groups=index_groups, mode=lime_mode
    )
    conditional_features_importance = {
        str(c): gdf for c, gdf in features_importance.groupby("Sample Group")
    }

    return LimeFeatureImportance(features_importance, conditional_features_importance)


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
        num_features = np.min([X.shape[1], 100])

    if num_samples is None:
        num_samples = np.min([X.shape[0], 1000])

    import random

    per_group_sample = int(np.ceil(num_samples / len(index_groups)))
    ids_groups = {
        str(label): random.sample(
            X.index[index].tolist(), min(len(X.index[index].tolist()), per_group_sample)
        )
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
    for label, indexes in tqdm(ids_groups.items()):
        for i in indexes:
            exp = explainer.explain_instance(
                X.loc[i], scorer, num_features=X.shape[1], num_samples=100
            )
            exp_values = list(exp.local_exp.values())[0]

            df_i = pd.DataFrame(exp_values, columns=["Feature Id", "Feature Weight"])
            df_i["Importance"] = df_i["Feature Weight"].abs()
            df_i["Importance"] = df_i["Importance"] / df_i["Importance"].sum()
            df_i["Sample Id"] = i
            df_i["Feature Label"] = X.columns[df_i["Feature Id"].tolist()]
            df_i["Feature Rank"] = range(1, df_i.shape[0] + 1)
            df_i["Sample Group"] = label
            df.append(df_i)

    df = pd.concat(df, axis=0, ignore_index=True)

    return df


class LimeFeatureImportance(BaseFeatureImportance, LocalFeatureImportance):
    def __init__(self, importance_weights, conditional_importance_weights):
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights

    def get_topk(self, top_k):
        if top_k is None:
            feat_imp = self.feature_importance
            cond_feat_imp = self.conditional_feature_importance
        else:
            feat_imp = get_top_k_lime(self.feature_importance, top_k)
            cond_feat_imp = {
                label: get_top_k_lime(value, top_k)
                for label, value in self.conditional_feature_importance.items()
            }

        return {
            "feature_importance": feat_imp,
            "conditional_feature_importance": cond_feat_imp,
        }

    def metrics(
        self, feature_importance, conditional_feature_importance, detailed=None
    ):

        reference_values = {
            "Features Spread Divergence": 0,
            "Features Spread Stability": 1,
            "Features Spread Mean": 0,
            "Dataset Spread Divergence": 0,
            "Dataset Spread Stability": 1,
            "Dataset Spread Mean": 0,
        }

        metrics = pd.concat(
            [
                dataset_spread_stability(
                    feature_importance, conditional_feature_importance
                )["result"],
                features_spread_stability(
                    feature_importance, conditional_feature_importance
                )["result"],
            ],
            axis=0,
        )

        reference_column = pd.DataFrame(
            [reference_values.get(metric) for metric in metrics.index],
            columns=["Reference"],
        ).set_index(metrics.index)
        metrics_with_reference = pd.concat([metrics, reference_column], axis=1)

        return metrics_with_reference

    def show_importance_stability(
        self, feature_importance, conditional_feature_importance
    ):

        import matplotlib.pyplot as plt
        import seaborn as sns

        data_stability = dataset_spread_stability(
            feature_importance, conditional_feature_importance
        )
        feature_stability = features_spread_stability(
            feature_importance, conditional_feature_importance
        )

        def format_data(d):
            df = []
            for g, x in d["imp_spread"].items():
                if not g == "Global":
                    a = pd.DataFrame(d["imp_spread"][g].copy())
                    a["Output"] = g
                    df.append(a)
            df = pd.concat(df, axis=0)
            df.columns = ["Importance Spread", "Output"]
            return df

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].set_title("Data Stability")
        df = format_data(data_stability)
        sns.boxplot(data=df, x="Importance Spread", y="Output", ax=axs[0])

        axs[1].set_title("Feature Stability")
        df = format_data(feature_stability)
        sns.boxplot(data=df, x="Importance Spread", y="Output", ax=axs[1])
        return data_stability, feature_stability