import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def contrast_matrix(xticks, values):   
    fig, ax = plt.subplots()
    fig.suptitle("Importance Constrast")
    cmap="Blues"

    ax = sns.heatmap(
        values,
        cbar=False,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=['Order','Range','Similarity'],
        xticklabels=xticks,
        cmap=cmap,
    )
    _ = plt.setp(ax.get_xticklabels(), fontsize=10, ha="center")
    _ = plt.setp(ax.get_yticklabels(), fontsize=10)
    
   
def partial_dependence_plot(x, features, title, model, grid_resolution=20, ax=None):
    from sklearn.inspection import PartialDependenceDisplay
    import matplotlib.pyplot as plt
        
    common_params = {
        "subsample": 50,
        "n_jobs": 2,
        "grid_resolution": grid_resolution,
        "random_state": 0,
    }
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    
    PartialDependenceDisplay.from_estimator(
            model,
            x,
            features,
            kind="average",
            ax=ax,
            **common_params)
    ax.set_title(title)    
    plt.show()