# base imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# utils
from holisticai.utils import get_colors

# range metrics
RANGE_METRICS_REGRESSION = {
    "RMSE Ratio": 0.2,
    "RMSE Ratio Q80": 0.2,
    "MAE Ratio": 0.2,
    "MAE Ratio Q80": 0.2,
    "Correlation Difference": 0.05,
    "Disparate Impact Q90": 0.05,
    "Disparate Impact Q80": 0.05,
    "Disparate Impact Q50": 0.05,
    "Statistical Parity Q50": 0.05,
    "Average Score Ratio": 0.05,
    "Average Score Difference": 0.1,
    "Z Score Difference": 0.1,
    "Max Statistical Parity": 0.1,
    "Statistical Parity AUC": 0.1,
}

RANGE_METRICS_CLASSIFICATION = {
    "Statistical Parity": 0.05,
    "Disparate Impact": 0.05,
    "Four Fifths Rule": 0.05,
    "Cohen D": 0.05,
    "2SD Rule": 0.05,
    "Equality of Opportunity Difference": 0.05,
    "False Positive Rate Difference": 0.05,
    "Average Odds Difference": 0.05,
    "Accuracy Difference": 0.05,
}

RANGE_METRICS_CLUSTERING = {
    "Cluster Balance": 0.05,
    "Minimum Cluster Ratio": 0.05,
    "Cluster Distribution Total Variation": 0.05,
    "Cluster Distribution KL Div": 0.05,
    "Social Fairness Ratio": 0.05,
    "Silhouette Difference": 0.05,
}


RANGE_METRICS = {
    "binary_classification": RANGE_METRICS_CLASSIFICATION,
    "regression": RANGE_METRICS_REGRESSION,
    "clustering": RANGE_METRICS_CLUSTERING,
}


def bias_metrics_report(
    model_type: str,
    table_metrics: pd.DataFrame,
    table_metrics_mitigated: pd.DataFrame = None,
):
    """
    Plot bias report for different model types.

    Parameters
    ----------
    model_type : str
        Type of model: 'binary_classification', 'regression', 'clustering'
    table_metrics : pandas.DataFrame
        Dataframe containing bias metrics.
    table_metrics_mitigated : bool, optional
        Whether the bias metrics are for mitigated model or not, by default False
    """
    metric_names = list(table_metrics.index)

    if table_metrics_mitigated is None:
        metrics_biased = table_metrics.copy()
        columns = ["Baseline", "Reference"]
        columns_plot = ["Baseline"]
        metrics_biased.columns = columns
        fill_range = [-1, 1]

    else:
        metrics_biased = pd.concat(
            [table_metrics["Value"], table_metrics_mitigated[["Value", "Reference"]]],
            axis=1,
        )
        columns = ["Baseline", "Mitigator", "Reference"]
        columns_plot = ["Baseline", "Mitigator"]
        metrics_biased.columns = columns
        fill_range = [-0.5, 1.5]

    cols = 4
    rows = len(metric_names) // cols
    if len(metric_names) % cols != 0:
        rows += 1

    max_cols_per_rows = 3
    fig_size = (12, 7) if rows >= max_cols_per_rows else (12, 4)

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=fig_size)

    for i, name in enumerate(metric_names):
        metric_data = metrics_biased[metrics_biased.index == name]
        row, col = divmod(i, cols)
        sns.barplot(
            data=metric_data[columns_plot],
            palette=get_colors(2),
            ax=axes[row, col],
        )
        axes[row, col].set_title(name)
        axes[row, col].axhline(y=metric_data["Reference"].values[0], color="black", linestyle="--")
        axes[row, col].set_ylabel("Score")

        if i == len(metric_names) - 1 and i % cols != cols - 1:
            for j in range(i % cols + 1, cols):
                axes[row, j].axis("off")

        elif i == len(metric_names) - 1 and i % cols == cols - 1:
            for j in range(cols):
                axes[row + 1, j].axis("off")

        if name != "No Disparate Impact Level":
            axes[row, col].fill_between(
                fill_range,
                metric_data["Reference"].values[0] - RANGE_METRICS[model_type][name],
                metric_data["Reference"].values[0] + RANGE_METRICS[model_type][name],
                color="slategray",
                alpha=0.4,
            )

        if i == 0:
            axes[row, col].legend(
                [
                    plt.Line2D([0], [0], linestyle="--", color="black", lw=2, label="Reference"),
                    plt.Rectangle(
                        [0, 0],
                        1,
                        1,
                        fc="slategray",
                        alpha=0.4,
                        ec="None",
                        label="Range",
                    ),
                ],
                ["Reference", "Fair Area"],
                loc="upper left",
                bbox_to_anchor=(-1, 1.0),
            )

        plt.tight_layout()
