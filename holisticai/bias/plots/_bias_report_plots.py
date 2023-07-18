# base imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# utils
from holisticai.utils import get_colors


def bias_report_regression(metrics: pd.DataFrame, metrics_mitigated: pd.DataFrame = None):
    """
    Plot bias report for regression models.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing bias metrics.
    mitigated : bool, optional
        Whether the bias metrics are for mitigated model or not, by default False
    """
    metric_names = list(metrics.index)

    cols = 4
    rows = len(metric_names) // cols
    if len(metric_names) % cols != 0:
        rows += 1

    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(15,7))

    if metrics_mitigated is None:
        for i, name in enumerate(metric_names):
            metrics_biased = metrics.copy()
            metrics_biased.columns = ['Baseline', 'Reference']
            metric_data = metrics_biased[metrics_biased.index == name]
            row, col = divmod(i, cols)
            axes[row, col].bar(metric_data.index, metric_data['Baseline'], label='Baseline', color=get_colors(1)[0])
            axes[row, col].set_title(name)
            axes[row, col].axhline(y=metric_data['Reference'].values[0], color='black', linestyle='--')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_xticklabels([])
            
            if i == len(metric_names) - 1 and i % cols != cols - 1:
                for j in range(i % cols + 1, cols):
                    axes[row, j].axis('off')
            elif i == len(metric_names) - 1 and i % cols == cols - 1:
                for j in range(cols):
                    axes[row + 1, j].axis('off')
            
            if name == 'RMSE Ratio':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'RMSE Ratio Q80':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'MAE Ratio':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'MAE Ratio Q80':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'Correlation Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q90':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q80':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q50':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Statistical Parity Q50':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Score Ratio':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Score Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Z Score Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Max Statistical Parity':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Statistical Parity AUC':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)

            if i == 0:
                axes[row, col].legend(['Reference', 'Fair',], bbox_to_anchor=(-0.85, 1), loc='upper left', fontsize=12)
            
            plt.tight_layout()
    
    else:
        for i, name in enumerate(metric_names):
            mitigated = pd.concat([metrics['Value'], metrics_mitigated[['Value', 'Reference']]], axis=1)
            mitigated.columns = ['Baseline', 'Mitigator', 'Reference']
            metric_data = mitigated[mitigated.index == name]
            row, col = divmod(i, cols)

            sns.barplot(data=metric_data[['Baseline', 'Mitigator']], palette=get_colors(2), ax=axes[row, col])
            axes[row, col].set_title(name)
            axes[row, col].axhline(y=metric_data['Reference'].values[0], color='black', linestyle='--')
            axes[row, col].set_ylabel('Score')
            
            if i == len(metric_names) - 1 and i % cols != cols - 1:
                for j in range(i % cols + 1, cols):
                    axes[row, j].axis('off')
            elif i == len(metric_names) - 1 and i % cols == cols - 1:
                for j in range(cols):
                    axes[row + 1, j].axis('off')
            
            if name == 'RMSE Ratio':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'RMSE Ratio Q80':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'MAE Ratio':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'MAE Ratio Q80':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.2, metric_data['Reference'].values[0] + 0.2, color='g', alpha=0.1)
            elif name == 'Correlation Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q90':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q80':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact Q50':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Statistical Parity Q50':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Score Ratio':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Score Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Z Score Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Max Statistical Parity':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)
            elif name == 'Statistical Parity AUC':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.1, metric_data['Reference'].values[0] + 0.1, color='g', alpha=0.1)

            if i == 0:
                axes[row, col].legend(['Baseline', 'Mitigator', 'Reference', 'Fair'], bbox_to_anchor=(-0.85, 1), loc='upper left', fontsize=12)  

            plt.tight_layout()

def bias_report_classification(metrics, metrics_mitigated = None):
    """
    Plot bias report for classification models.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing bias metrics.
    mitigated : bool, optional
        Whether the bias metrics are for mitigated model or not, by default False
    """
    metric_names = list(metrics.index)

    cols = 4
    rows = len(metric_names) // cols
    if len(metric_names) % cols != 0:
        rows += 1

    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(15,8))
    if metrics_mitigated is None:
        for i, name in enumerate(metric_names):
            metrics_biased = metrics.copy()
            metrics_biased.columns = ['Baseline', 'Reference']
            metric_data = metrics_biased[metrics_biased.index == name]
            row, col = divmod(i, cols)
            axes[row, col].bar(metric_data.index, metric_data['Baseline'], label='Baseline', color=get_colors(1)[0])
            axes[row, col].set_title(name)
            axes[row, col].axhline(y=metric_data['Reference'].values[0], color='black', linestyle='--')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_xticklabels([])
            
            if i == len(metric_names) - 1 and i % cols != cols - 1:
                for j in range(i % cols + 1, cols):
                    axes[row, j].axis('off')
            elif i == len(metric_names) - 1 and i % cols == cols - 1:
                for j in range(cols):
                    axes[row + 1, j].axis('off')
            
            if name == 'Statistical Parity':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Four Fifths Rule':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Equality of Opportunity Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'False Positive Rate Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Odds Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Accuracy Difference':
                axes[row, col].fill_between([-1, 1], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)

            if i == 0:
                axes[row, col].legend(['Reference', 'Fair',], bbox_to_anchor=(-0.85, 1), loc='upper left', fontsize=12)

            plt.tight_layout()

    else:
        for i, name in enumerate(metric_names):
            mitigated = pd.concat([metrics['Value'], metrics_mitigated[['Value', 'Reference']]], axis=1)
            mitigated.columns = ['Baseline', 'Mitigator', 'Reference']
            metric_data = mitigated[mitigated.index == name]
            
            row, col = divmod(i, cols)

            sns.barplot(data=metric_data[['Baseline', 'Mitigator']], palette=get_colors(2), ax=axes[row, col])
            axes[row, col].set_title(name)
            axes[row, col].axhline(y=metric_data['Reference'].values[0], color='black', linestyle='--')
            axes[row, col].set_ylabel('Score')
            
            if i == len(metric_names) - 1 and i % cols != cols - 1:
                for j in range(i % cols + 1, cols):
                    axes[row, j].axis('off')
            elif i == len(metric_names) - 1 and i % cols == cols - 1:
                for j in range(cols):
                    axes[row + 1, j].axis('off')
            
            if name == 'Statistical Parity':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Disparate Impact':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Four Fifths Rule':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Equality of Opportunity Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'False Positive Rate Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Average Odds Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)
            elif name == 'Accuracy Difference':
                axes[row, col].fill_between([-0.5, 1.5], metric_data['Reference'].values[0] - 0.05, metric_data['Reference'].values[0] + 0.05, color='g', alpha=0.1)

            if i == 0:
                axes[row, col].legend(['Baseline', 'Mitigator', 'Reference', 'Fair'], bbox_to_anchor=(-0.85, 1), loc='upper left', fontsize=12)

            plt.tight_layout()