import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..local_importance._local_metrics import DataStability, FeatureStability
from ..utils import (
    BaseFeatureImportance,
    LocalFeatureImportance,
    check_feature_importance,
    get_alpha_lime,
    get_index_groups,
)


def compute_local_feature_importance(
    model_type, x, y, local_explainer_handler, num_samples=1000
):
    features_importance = compute_local_importants(
        model_type, x, y, local_explainer_handler, num_samples
    )
    conditional_features_importance = {
        str(c): gdf for c, gdf in features_importance.groupby("Sample Group")
    }
    return TabularLocalFeatureImportance(
        features_importance, conditional_features_importance
    )


def grouped_sample(X, index_groups, num_samples=1000):
    num_samples = np.min([X.shape[0], num_samples])
    import random

    per_group_sample = int(np.ceil(num_samples / len(index_groups)))
    ids_groups = {
        str(label): random.sample(list(index), min(len(index), per_group_sample))
        for label, index in index_groups.items()
    }
    return ids_groups


def stratified_sample(model_type, X, y, num_samples=1000):
    group2index = get_index_groups(model_type, y)
    group2index = grouped_sample(X, group2index, num_samples=num_samples)
    indexes = [idx for indexes in group2index.values() for idx in indexes]
    Xsel = X.loc[indexes]
    return Xsel, {i: k for k, ids in group2index.items() for i in ids}


def compute_local_importants(model_type, X, y, local_explainer, num_samples=1000):
    Xsel, index2group = stratified_sample(model_type, X, y, num_samples=num_samples)
    imp = local_explainer(Xsel)
    rank = pd.DataFrame(
        imp.values.argsort(axis=1) + 1, index=imp.index, columns=imp.columns
    )
    feat_names = imp.columns
    feat2id = {f: i for i, f in enumerate(feat_names)}
    df_imp = (
        pd.melt(imp, value_vars=feat_names, ignore_index=False)
        .rename({"variable": "Feature Label", "value": "Importance"}, axis=1)
        .reset_index(names=["Sample Id"])
    )
    df_rank = (
        pd.melt(rank, value_vars=feat_names, ignore_index=False)
        .rename({"variable": "Feature Label", "value": "Feature Rank"}, axis=1)
        .reset_index(names=["Sample Id"])
    )
    df = pd.merge(
        left=df_imp, right=df_rank, how="inner", on=["Sample Id", "Feature Label"]
    )
    df["Feature Id"] = df["Feature Label"].apply(lambda x: feat2id[x])
    df["Sample Group"] = df["Sample Id"].apply(lambda x: index2group[x])
    return df


class TabularLocalFeatureImportance(BaseFeatureImportance, LocalFeatureImportance):
    def __init__(self, importance_weights, conditional_importance_weights):
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights

    def get_alpha_feature_importance(self, alpha):
        if alpha is None:
            feat_imp = self.feature_importance
            cond_feat_imp = self.conditional_feature_importance
        else:
            feat_imp = get_alpha_lime(self.feature_importance, alpha)
            cond_feat_imp = {
                label: get_alpha_lime(value, alpha)
                for label, value in self.conditional_feature_importance.items()
            }

        return feat_imp, cond_feat_imp

    def metrics(self, alpha=None, detailed=False):

        fs = FeatureStability(detailed=detailed)
        scores = fs(self.feature_importance, self.conditional_feature_importance)

        metric_scores = []
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Referemce": fs.reference}
            for metric_name, value in scores.items()
        ]

        ds = DataStability(detailed=detailed)
        scores = ds(self.feature_importance, self.conditional_feature_importance)
        metric_scores += [
            {"Metric": metric_name, "Value": value, "Referemce": fs.reference}
            for metric_name, value in scores.items()
        ]

        return pd.DataFrame(metric_scores).set_index("Metric").sort_index()

    def show_importance_stability(
        self, feature_importance, conditional_feature_importance
    ):

        import matplotlib.pyplot as plt
        import seaborn as sns

        fs = FeatureStability(detailed=True)
        ds = DataStability(detailed=True)

        data_stability = ds(
            feature_importance, conditional_feature_importance, reduce=False
        )
        feature_stability = fs(
            feature_importance, conditional_feature_importance, reduce=False
        )

        def format_data(name, d):
            df = []
            for g, x in d.items():
                a = pd.DataFrame(d[g].copy())
                a["Output"] = g
                df.append(a)
            df = pd.concat(df, axis=0)
            df.columns = [name, "Output"]
            return df

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].set_title(ds.name)
        df = format_data("Spread Ratio", data_stability)
        sns.boxplot(data=df, x="Spread Ratio", y="Output", ax=axs[0])
        axs[0].grid()

        axs[1].set_title(fs.name)
        df = format_data("Spread Ratio", feature_stability)
        sns.boxplot(data=df, x="Spread Ratio", y="Output", ax=axs[1])
        axs[1].grid()

    def show_data_stability_boundaries(self, n_rows, n_cols, top_n=None, figsize=None):
        if figsize is None:
            figsize = (15, 5)

        if top_n is None:
            top_n = 10

        fimp, cfimp = self.get_alpha_feature_importance(None)
        ds = DataStability(detailed=True)
        spread = ds(fimp, cfimp, reduce=False)

        fimp = fimp.groupby("Feature Label")["Importance"].mean()

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

        def show_importance(feature_importance, index, ax):
            Q = feature_importance
            qs = Q[Q["Sample Id"] == index]
            qs.set_index("Feature Label")["Importance"].sort_values(
                ascending=True
            ).iloc[-top_n:].plot(kind="barh", color="mediumslateblue", ax=ax)
            return qs["Importance"].max()

        i = 0
        min_values = []
        max_values = []
        for g, _ in cfimp.items():
            s = spread[f"{ds.name} {g}"]
            max_index = s.idxmax()
            min_index = s.idxmin()

            min_values.append(show_importance(cfimp[g], min_index, axs[0, i]))
            axs[0, i].set_title(f"{g} Min Ratio [{s.loc[min_index]:.3f}]")

            max_values.append(show_importance(cfimp[g], max_index, axs[1, i]))
            axs[1, i].set_title(f"{g} Max Ratio [{s.loc[max_index]:.3f}]")

            i += 1

        i = 0
        xlim0 = max(min_values)
        xlim1 = max(max_values)
        xlim = max([xlim0, xlim1])
        for g, s in cfimp.items():
            axs[0, i].set_xlim([0, xlim])
            axs[1, i].set_xlim([0, xlim])
            axs[0, i].grid(True)
            axs[1, i].grid(True)
            i += 1
        fig.tight_layout()

    def show_features_stability_boundaries(self, n_rows, n_cols, figsize=None):

        if figsize is None:
            figsize = (15, 5)

        fimp, cfimp = self.get_alpha_feature_importance(alpha=None)
        fimp = fimp.dropna()
        cfimp = {k: v.dropna() for k, v in cfimp.items()}

        fs = FeatureStability(detailed=True)
        spread = fs(fimp, cfimp, reduce=False)

        if n_cols is None:
            n_cols = len(spread)
            n_rows = 2

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
        max_values = []
        for i, (g, _) in enumerate(cfimp.items()):
            s = spread[f"{fs.name} {g}"]
            min_index = s.idxmin()
            max_index = s.idxmax()

            fi = cfimp[g]

            importances = fi[fi["Feature Label"] == min_index]["Importance"]
            max_value1 = importances.max()
            importances.plot(kind="hist", ax=axs[0][i], color="mediumslateblue")
            axs[0][i].set_title(f"{g} R[{min_index}]= {s.loc[min_index]:.3f}")
            axs[0][i].set_xlabel("Importance")

            importances = fi[fi["Feature Label"] == max_index]["Importance"]
            importances.plot(kind="hist", ax=axs[1][i], color="mediumslateblue")
            max_value2 = importances.max()
            axs[1][i].set_title(f"{g} R[{max_index}]= {s.loc[max_index]:.3f}")
            axs[1][i].set_xlabel("Importance")

            max_values.append(max([max_value1, max_value2]))

        i = 0
        xlim = max(max_values)
        for g, _ in cfimp.items():
            axs[0, i].set_xlim([0, xlim])
            axs[1, i].set_xlim([0, xlim])
            axs[0, i].grid(True)
            axs[1, i].grid(True)
            i += 1
        fig.tight_layout()
