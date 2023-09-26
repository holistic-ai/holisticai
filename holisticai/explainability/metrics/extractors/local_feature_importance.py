import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from holisticai.explainability.metrics.local_importance._local_metrics import (
    dataset_spread_stability,
    features_spread_stability,
)

from ..utils import (
    check_feature_importance,
    get_index_groups,
    get_top_k_lime,
    BaseFeatureImportance,
    LocalFeatureImportance
)

from ..local_importance._local_metrics import (
    dataset_spread_stability,
    features_spread_stability,
)

def compute_local_feature_importance(model_type, x, y, local_explainer_handler, num_samples=1000):
    features_importance = compute_local_importants(model_type, x, y, local_explainer_handler, num_samples)
    conditional_features_importance = {
        str(c): gdf for c, gdf in features_importance.groupby("Sample Group")
    }
    return TabularLocalFeatureImportance(features_importance, conditional_features_importance)

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
    return Xsel , {i:k for k,ids in group2index.items() for i in ids}

def compute_local_importants(model_type, X, y, local_explainer, num_samples=1000):
    Xsel, index2group = stratified_sample(model_type, X, y, num_samples=num_samples)
    imp = local_explainer(Xsel)
    rank = pd.DataFrame(imp.values.argsort(axis=1)+1, index=imp.index, columns=imp.columns)
    feat_names = imp.columns
    feat2id = {f:i for i,f in enumerate(feat_names)}
    df_imp = pd.melt(imp, value_vars=feat_names, ignore_index=False).rename({'variable':'Feature Label', 'value':'Importance'}, axis=1).reset_index(names=['Sample Id'])
    df_rank = pd.melt(rank, value_vars=feat_names, ignore_index=False).rename({'variable':'Feature Label', 'value':'Feature Rank'}, axis=1).reset_index(names=['Sample Id'])
    df = pd.merge(left=df_imp, right=df_rank, how='inner', on=['Sample Id', 'Feature Label'])
    df['Feature Id'] = df['Feature Label'].apply(lambda x:feat2id[x])
    df["Sample Group"] = df["Sample Id"].apply(lambda x:index2group[x])
    return df
    
class TabularLocalFeatureImportance(BaseFeatureImportance, LocalFeatureImportance):
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
        self, feature_importance, conditional_feature_importance, detailed=False
    ):

        reference_values = {
            "Features Stability Gini": 0,
            "Features Stability Ratio": 1,
            "Features Stability Mean": 0,
            "Dataset Stability Gini": 0,
            "Dataset Stability Ratio": 1,
            "Dataset Stability Mean": 0,
        }

        if not detailed:
            d_spread_stability = dataset_spread_stability(
                feature_importance, conditional_feature_importance
            )["result"]
            d_spread_stability = {k: v["Global"] for k, v in d_spread_stability.items()}
            d_spread_stability = pd.DataFrame(d_spread_stability, index=[0])
            d_spread_stability = d_spread_stability.T.rename(columns={0: "Value"})

            f_spread_stability = features_spread_stability(
                feature_importance, conditional_feature_importance
            )["result"]
            f_spread_stability = {k: v["Global"] for k, v in f_spread_stability.items()}
            f_spread_stability = pd.DataFrame(f_spread_stability, index=[0])
            f_spread_stability = f_spread_stability.T.rename(columns={0: "Value"})
        else:
            def rename_metric(x):
                if not (x['variable']=='Global'):
                    return f"{x['index']} {x['variable']}"
                return f"{x['index']}"
            
            d_spread_stability = dataset_spread_stability(
                feature_importance, conditional_feature_importance
            )["result"]
            groups = list(d_spread_stability["Dataset Stability Gini"].keys())
            d_spread_stability = pd.DataFrame(d_spread_stability).T.reset_index()
            d_spread_stability = pd.melt(
                d_spread_stability, id_vars=["index"], value_vars=groups
            ).reset_index()
            d_spread_stability["Metric"] = d_spread_stability.apply(
                lambda x: rename_metric(x), axis=1
            )
            d_spread_stability.sort_values("index", inplace=True)
            d_spread_stability = d_spread_stability[["Metric", "value"]].set_index(
                "Metric"
            )
            d_spread_stability = d_spread_stability.rename(
                columns={"value": "Value"}
            )

            f_spread_stability = features_spread_stability(
                feature_importance, conditional_feature_importance
            )["result"]
            groups = list(f_spread_stability["Features Stability Gini"].keys())
            f_spread_stability = pd.DataFrame(f_spread_stability).T.reset_index()
            f_spread_stability = pd.melt(
                f_spread_stability, id_vars=["index"], value_vars=groups
            ).reset_index()
            
            f_spread_stability["Metric"] = f_spread_stability.apply(
                lambda x: rename_metric(x), axis=1
            )
            f_spread_stability.sort_values("index", inplace=True)
            f_spread_stability = f_spread_stability[["Metric", "value"]].set_index(
                "Metric"
            )
            f_spread_stability = f_spread_stability.rename(
                columns={"value": "Value"}
            )

        metrics = pd.concat([d_spread_stability, f_spread_stability], axis=0)

        def remove_label_markers(metric):
            words = metric.split(" ")
            if words[-1] == "Global":
                metric = " ".join([w for w in words[:-1]])
            else:
                metric = " ".join([w for w in words if not w.startswith("[")])
            return metric

        reference_column = pd.DataFrame(
            [
                reference_values.get(
                    metric, reference_values[remove_label_markers(metric)]
                )
                for metric in metrics.index
            ],
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

        metric_name = "Importance Spread Ratio"

        def format_data(d):
            df = []
            for g, x in d["imp_spread"].items():
                if not g == "Global":
                    a = pd.DataFrame(d["imp_spread"][g].copy())
                    a["Output"] = g
                    df.append(a)
            df = pd.concat(df, axis=0)
            df.columns = [metric_name, "Output"]
            return df

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].set_title("Data Stability")
        df = format_data(data_stability)
        sns.boxplot(data=df, x=metric_name, y="Output", ax=axs[0])
        axs[0].grid()
        
        axs[1].set_title("Feature Stability")
        df = format_data(feature_stability)
        sns.boxplot(data=df, x=metric_name, y="Output", ax=axs[1])
        axs[1].grid()

    def show_data_stability_boundaries(self, top_n=None, figsize=None):
        if figsize is None:
            figsize = (15, 5)

        if top_n is None:
            top_n = 10

        all_fimp = self.get_topk(None)
        data_stability = dataset_spread_stability(**all_fimp)

        spread = data_stability["imp_spread"]
        cfimp = all_fimp["conditional_feature_importance"]
        fimp = (
            all_fimp["feature_importance"].groupby("Feature Label")["Importance"].mean()
        )

        fig, axs = plt.subplots(len(spread) - 1, 2, figsize=figsize)

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
        for g, s in spread.items():
            max_index = s.idxmax()
            min_index = s.idxmin()

            if not (g == "Global"):
                min_values.append(show_importance(cfimp[g], min_index, axs[i, 0]))
                axs[i, 0].set_title(f"{g} Min Ratio [{s.loc[min_index]:.3f}]")

                max_values.append(show_importance(cfimp[g], max_index, axs[i, 1]))
                axs[i, 1].set_title(f"{g} Max Ratio [{s.loc[max_index]:.3f}]")

                i += 1

        i = 0
        xlim0 = max(min_values)
        xlim1 = max(max_values)
        xlim = max([xlim0, xlim1])
        for g, s in spread.items():
            if not (g == "Global"):
                axs[i, 0].set_xlim([0, xlim])
                axs[i, 1].set_xlim([0, xlim])
                axs[i, 0].grid(True)
                axs[i, 1].grid(True)
                i += 1
        fig.tight_layout()

    def show_features_stability_boundaries(self, figsize=None):
        from holisticai.explainability.metrics.local_importance._local_metrics import (
            features_spread_stability,
        )

        if figsize is None:
            figsize = (15, 5)

        all_fimp = self.get_topk(None)
        feature_stability = features_spread_stability(**all_fimp)

        spread = feature_stability["imp_spread"]
        cfimp = all_fimp["conditional_feature_importance"]
        fimp = all_fimp["feature_importance"]
        fig, axs = plt.subplots(len(spread), 2, figsize=figsize)
        max_values = []
        for i, (g, s) in enumerate(spread.items()):
            min_index = s.idxmin()
            max_index = s.idxmax()

            if not (g == "Global"):
                fi = cfimp[g]
            else:
                fi = fimp

            importances = fi[fi["Feature Label"] == min_index]["Importance"]
            max_value1 = importances.max()
            importances.plot(kind="hist", ax=axs[i][0])
            axs[i][0].set_title(f"{g} R[{min_index}]= {s.loc[min_index]:.3f}")
            axs[i][0].set_xlabel("Importance")

            importances = fi[fi["Feature Label"] == max_index]["Importance"]
            importances.plot(kind="hist", ax=axs[i][1])
            max_value2 = importances.max()
            axs[i][1].set_title(f"{g} R[{max_index}]= {s.loc[max_index]:.3f}")
            axs[i][1].set_xlabel("Importance")

            max_values.append(max([max_value1, max_value2]))

        i = 0
        xlim = max(max_values)
        for g, s in spread.items():
            axs[i, 0].set_xlim([0, xlim])
            axs[i, 1].set_xlim([0, xlim])
            axs[i, 0].grid(True)
            axs[i, 1].grid(True)
            i += 1
        fig.tight_layout()
