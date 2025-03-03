import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors


def local_importances_to_numpy(local_importances, ranked_columns=None):
    if ranked_columns is None:
        local_importances = local_importances.data["DataFrame"].to_numpy()
    else:
        local_importances = local_importances.data["DataFrame"][ranked_columns].to_numpy()
    # print(local_importances)
    local_importances /= local_importances.sum(axis=1, keepdims=True)
    return local_importances


def compute_importance_distribution(local_importances, k=5, num_samples=10000, random_state=None):
    if isinstance(random_state, int):
        random_state = RandomState(random_state)
    if random_state is None:
        random_state = RandomState(42)

    num_samples = 10000
    embeddings = local_importances_to_numpy(local_importances)

    num_features = embeddings.shape[1]
    alpha = 0.1
    grid = np.array([random_state.dirichlet(alpha * np.ones(num_features)) for _ in range(num_samples)])
    nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
    distances, _ = nbrs.kneighbors(grid)
    densities = 1 / (distances[:, -1] + 1e-10)  # Evitar división por 0
    densities /= densities.sum()
    return densities


def feature_stability(local_feature_importance, strategy="variance", **kargs):
    fs = FeatureStability()
    return fs(local_feature_importance, strategy=strategy, **kargs)


class FeatureStability:
    reference: int = 1
    name: str = "Feature Stability"

    def __call__(self, local_feature_importance, strategy="variance", **kargs):
        densities = compute_importance_distribution(local_feature_importance, **kargs)
        if strategy == "variance":
            jsd_std = np.std(densities)
            jsd_max = np.max(densities)
            return 1 - (jsd_std / jsd_max)
        if strategy == "entropy":
            num_samples = len(densities)
            feature_equal_weight = np.array([1.0 / num_samples] * num_samples)
            return 1 - jensenshannon(densities, feature_equal_weight, base=2)
        msg = f"Invalid strategy: {strategy}"
        raise ValueError(msg)
