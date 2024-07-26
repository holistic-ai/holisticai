import numpy as np
import matplotlib.pyplot as plt
from holisticai.explainability.plots import plot_partial_dependence,plot_feature_importance
from holisticai.utils import PartialDependence, Importances

static_image_folder = '_static/images'
def image_plot_partial_dependence():
    noise = 0.1*np.random.randn(50,5)
    partial_dependence = PartialDependence(values = [[
        {
            "individual": np.array([[[1, 2, 3, 3.7, 4.5]]])+noise,
            "average": np.array([[1, 2, 3, 3.7 ,4.5]]),
            "grid_values": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        }
    ]])
    ranked_feature_importance = Importances(values=[0.123], feature_names=["Feature 1"])

    plot_partial_dependence(partial_dependence, ranked_feature_importance)
    plt.savefig(f"{static_image_folder}/xai_plot_partial_dependence.png")
    
def image_plot_feature_importance():
    feature_importance = Importances(values=np.array([0.1, 0.2, 0.3, 0.4]), feature_names=['A', 'B', 'C', 'D'])
    fig,ax = plt.subplots(1,1)
    ax = plot_feature_importance(feature_importance, ax=ax)
    fig.savefig(f"{static_image_folder}/xai_plot_feature_importance.png")


def image_plot_surrogate():
    from holisticai.explainability.plots import plot_surrogate
    from holisticai.utils import Importances
    from holisticai.datasets import load_dataset
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt

    ds = load_dataset("adult")
    surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate.fit(ds['X'], ds['y'])

    importance = Importances(values=surrogate.feature_importances_, feature_names=ds['X'].columns, extra_attrs={'surrogate': surrogate})
    fig,ax = plt.subplots(1,1, figsize=(20, 10))
    plot_surrogate(importance, ax)
    fig.savefig(f"{static_image_folder}/xai_plot_surrogate.png")