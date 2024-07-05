import errno
import logging
import os
import os.path as osp
import timeit

import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalizefea(X):
    """
    L2 normalize
    """
    feanorm = np.maximum(1e-14, np.sum(X**2, axis=1))
    X_out = X / (feanorm[:, None] ** 0.5)
    return X_out


def get_V_jl(x, L, N, K):  # noqa: N802
    x = x.squeeze()
    temp = np.zeros((N, K))
    index_cluster = L[x]
    temp[(x, index_cluster)] = 1
    temp = temp.sum(0)
    return temp


def get_fair_accuracy(group_prob, groups_ids, L, K):
    N = len(groups_ids)
    V_j_list = np.array([get_V_jl(groups_ids[f"{g}"], L, N, K) for g in group_prob.index])

    balance = np.zeros(K)
    J = len(group_prob)
    for k in range(K):
        V_j_list_k = V_j_list[:, k].copy()
        balance_temp = np.tile(V_j_list_k, [J, 1])
        balance_temp = balance_temp.T / np.maximum(balance_temp, 1e-20)
        mask = np.ones(balance_temp.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        balance[k] = balance_temp[mask].min()

    return balance.min(), balance.mean()


def get_fair_accuracy_proportional(group_prob, groups_ids, L, K):
    N = len(groups_ids)
    V_j_list = np.array([get_V_jl(groups_ids[f"{g}"], L, N, K) for g in group_prob.index])
    clustered_uV = V_j_list / sum(V_j_list)
    fairness_error = np.zeros(K)
    u_V = np.array(group_prob)
    for k in range(K):
        fairness_error[k] = (-u_V * np.log(np.maximum(clustered_uV[:, k], 1e-20)) + u_V * np.log(u_V)).sum()
    return fairness_error.sum()


def create_affinity(X, knn, scale=None, savepath=None, W_path=None):
    N, D = X.shape
    if W_path is not None:
        if W_path.endswith(".mat"):
            W = sio.loadmat(W_path)["W"]
        elif W_path.endswith(".npz"):
            W = sparse.load_npz(W_path)
    else:
        logger.info("Compute Affinity ")
        start_time = timeit.default_timer()

        nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), knn - 1)
        col = knnind[:, 1:].flatten()
        if scale is None:
            data = np.ones(X.shape[0] * (knn - 1))
        elif scale is True:
            scale = np.median(dist[:, 1:])
            data = np.exp((-(dist[:, 1:] ** 2)) / (2 * scale**2)).flatten()
        else:
            data = np.exp((-(dist[:, 1:] ** 2)) / (2 * scale**2)).flatten()

        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=float)
        W = (W + W.transpose(copy=True)) / 2
        elapsed = timeit.default_timer() - start_time
        logger.info(elapsed)

        if isinstance(savepath, str):
            if savepath.endswith(".npz"):
                sparse.save_npz(savepath, W)
            elif savepath.endswith(".mat"):
                sio.savemat(savepath, {"W": W})

    return W
