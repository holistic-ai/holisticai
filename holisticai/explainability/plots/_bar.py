import matplotlib.pyplot as plt
import seaborn as sns

from holisticai.utils import get_colors


def bar(
    feat_imp,
    top_k_sep,
    max_display=6,
    title=None,
    figsize=(7, 5),
    _type="global",
    ax=None,
):
    """
    Parameters
    ----------
    max_display: int
        Maximum number of features to display
    title: str
        Title of the plot
    top_k: int
        Number of features to select
    figsize: tuple
        Size of the plot
    _type: str
        Type of feature importance
    """
    if max_display is None:
        max_features = feat_imp.shape[0]
    else:
        max_features = max_display

    if _type == "global":
        feat_imp, _ = feat_imp
        feat_imp = feat_imp.reset_index("Variable")
        df_feat_imp = feat_imp.iloc[0:max_features, :].sort_values(
            by="Importance", ascending=False
        )
        importance = df_feat_imp["Importance"]
        df_feat_imp["Variable"] = df_feat_imp["Variable"].str[:max_display]
        names = df_feat_imp["Variable"]

    else:
        feat_imp["Feature Label"] = feat_imp["Feature Label"].str[:max_display]
        df_feat_imp = (
            feat_imp.groupby("Feature Label")["Importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        df_feat_imp = df_feat_imp.iloc[0:max_features, :]
        names = df_feat_imp["Feature Label"]

    importance = df_feat_imp["Importance"]

    colors = get_colors(10)
    hai_palette = sns.color_palette(colors)

    sns.set(style="whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        y=names,
        x=importance,
        palette=hai_palette,
        ax=ax,
    )
    ax.axhline(y=top_k_sep - 0.5, color="red", linewidth=2)
    ax.set_title(title)
    plt.tight_layout()
