import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances as ecdist


def reduce_func(D_chunk, start):
    J = np.mean(D_chunk, axis=1)
    return J


def NormalizedCutEnergy(A, S, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):

        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    nassoc_e = 0
    num_cluster = 0
    for k in range(maxclusterid + 1):
        S_k = S[:, k]
        # print S_k
        if 0 == np.sum(clustering == k):
            continue  # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot(np.dot(np.transpose(S_k), A), S_k) / np.dot(
                np.transpose(d), S_k
            )
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(
                np.transpose(d), S_k
            )
            nassoc_e = nassoc_e[0, 0]
    ncut_e = num_cluster - nassoc_e

    return ncut_e


def NormalizedCutEnergy_discrete(A, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):

        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    nassoc_e = 0
    num_cluster = 0
    for k in range(maxclusterid + 1):
        S_k = np.array(clustering == k, dtype=np.float)
        # print S_k
        if 0 == np.sum(clustering == k):
            continue  # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot(np.dot(np.transpose(S_k), A), S_k) / np.dot(
                np.transpose(d), S_k
            )
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(
                np.transpose(d), S_k
            )
            nassoc_e = nassoc_e[0, 0]
    ncut_e = num_cluster - nassoc_e

    return ncut_e


def KernelBound_k(A, d, S_k, N):
    # S_k = S[:,k]
    volume_s_k = np.dot(np.transpose(d), S_k)
    volume_s_k = volume_s_k[0, 0]
    temp = np.dot(np.transpose(S_k), A.dot(S_k)) / (volume_s_k * volume_s_k)
    temp = temp * d
    temp2 = temp + np.reshape(-2 * A.dot(S_k) / volume_s_k, (N, 1))

    return temp2.flatten()


def km_le(X, M):

    """
    Discretize the assignments based on center

    """
    e_dist = ecdist(X, M)
    l = e_dist.argmin(axis=1)

    return l


# Fairness term calculation
def fairness_term_V_j(u_j, S, V_j):
    V_j = V_j.astype("float")
    S_term = np.maximum(np.dot(V_j, S), 1e-20)
    S_sum = np.maximum(S.sum(0), 1e-20)
    S_term = u_j * (np.log(S_sum) - np.log(S_term))
    return S_term


def km_discrete_energy(e_dist, l, k):
    tmp = np.asarray(np.where(l == k)).squeeze()
    return np.sum(e_dist[tmp, k])


def compute_fairness_energy(group_prob, groups_ids, S, bound_lambda):
    """
    compute fair clustering energy
    """
    fairness_E = [
        fairness_term_V_j(group_prob.loc[g], S, groups_ids[f"{g}"])
        for g in group_prob.index
    ]
    fairness_E = (bound_lambda * sum(fairness_E)).sum()
    return fairness_E


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)


def _init_centroids(
    X, n_clusters, init, random_state=np.random.RandomState(42), init_size=None
):

    from sklearn.cluster._kmeans import _kmeans_plusplus
    from sklearn.utils.validation import check_array

    x_squared_norms = np.linalg.norm(X, axis=1)
    n_samples = X.shape[0]
    # n_clusters = n_clusters if n_centroids is None else n_centroids

    if init_size is not None and init_size < n_samples:
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]

    if isinstance(init, str) and init == "k-means++":
        centers, _ = _kmeans_plusplus(
            X,
            n_clusters,
            random_state=random_state,
            x_squared_norms=x_squared_norms,
        )
    elif isinstance(init, str) and init == "random":
        seeds = random_state.permutation(n_samples)[:n_clusters]
        centers = X[seeds]
    elif _is_arraylike_not_scalar(init):
        centers = init
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
        # _validate_center_shape(X, centers)
    import scipy.sparse as sp

    if sp.issparse(centers):
        centers = centers.toarray()

    return centers
