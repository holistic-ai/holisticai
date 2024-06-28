from matplotlib import pyplot as plt


def plot_feature_importance(feature_importance, ranked_feature_importance, top_n=20, ax=None):
    ranked_feature_importance = ranked_feature_importance.feature_importances.set_index("Variable")
    feature_importances = feature_importance.feature_importances.set_index("Variable")
    feature_importances.loc[:, "color"] = "#21918C"
    feature_importances.loc[ranked_feature_importance.index, "color"] = "#440154"
    feature_importances.reset_index(inplace=True, drop=False)

    top_n = min(top_n, len(feature_importances))
    top_features = feature_importances.sort_values(by="Importance", ascending=True).tail(top_n)
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax = top_features.plot(kind="barh", x="Variable", y="Importance", color=top_features["color"], legend=False, ax=ax)
    ax.axhline(y=len(top_features) - len(ranked_feature_importance) - 0.5, color="red", linestyle="--", linewidth=2)
    ax.grid()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")

    if hasattr(feature_importance, "strategy"):
        ax.set_title(f"{feature_importance.strategy.title()} Feature Importance")
    else:
        ax.set_title("Feature Importance")
