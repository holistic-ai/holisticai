import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def contrast_matrix(xticks, values, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Importance Constrast")
    cmap = "Blues"

    sns.heatmap(
        values,
        cbar=False,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=["Order", "Range", "Similarity"],
        xticklabels=xticks,
        cmap=cmap,
        ax=ax
    )
    _ = plt.setp(ax.get_xticklabels(), fontsize=10, ha="center")
    _ = plt.setp(ax.get_yticklabels(), fontsize=10)
