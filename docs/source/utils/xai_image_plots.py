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


from holisticai.robustness.plots._dataset_shift import (
    plot_2d,
    plot_adp_and_adf,
    plot_label_and_prediction,
    plot_neighborhood,
)

def image_plot_dataset_shift():
    
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn import tree

    # Pre-Plots
    # ---------

    center_box = 2.5
    X, y = make_blobs(n_samples=100, 
                    centers=2, 
                    n_features=2, 
                    cluster_std=0.8, 
                    center_box=(-center_box, center_box), 
                    random_state=42)
    
    # Array of indices
    indices = np.arange(X.shape[0])

    # Split the indices
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)

    # Split the data using the indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Train a classifier over the data
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Plots
    # -----

    # Scatter Plot of a 2D dataset
    plot_2d(X, y)
    plt.savefig(f"{static_image_folder}/plot_2d_pure.png")

    # Scatter Plot of a 2D dataset with a highlighted group
    plot_2d(X, y, highlight_group=test_indices)
    plt.savefig(f"{static_image_folder}/plot_2d_highlight_group.png")

    # Scatter Plot of a 2D dataset with a highlighted group and it's labels
    plot_2d(X, y, highlight_group=test_indices, show_just_group=True)
    plt.savefig(f"{static_image_folder}/plot_2d_show_just_group.png")

    # Scatter Plot of a 2D dataset with y_test and y_pred together in the same graph
    plot_label_and_prediction(X_test, y_test, y_pred, vertical_offset=0.1)
    plt.savefig(f"{static_image_folder}/plot_2d_label_and_prediction.png")

    # Scatter Plot of a 2D dataset with y_test and y_pred together with neighborhood accuracy calculation
    plot_neighborhood(X_test, y_test, y_pred, n_neighbors=4, points_of_interest=[13, 16, 19], vertical_offset=0.1)
    plt.savefig(f"{static_image_folder}/plot_2d_neighborhood.png")
