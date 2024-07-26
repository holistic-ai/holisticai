import numpy as np
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._bound_update import (
    get_S_discrete,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._method_utils import (
    KernelBound_k,
    NormalizedCutEnergy,
    NormalizedCutEnergy_discrete,
    ecdist,
    km_discrete_energy,
    km_le,
    reduce_func,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._utils import create_affinity
from sklearn.metrics import pairwise_distances_chunked as pdist_chunk


class KUtils:
    def __init__(self, X, K):
        self.N = len(X)
        self.K = K
        self.X = X

    def update(self, a_p, L, C):  # noqa: ARG002
        S = get_S_discrete(L, self.N, self.K)
        L = km_le(self.X, C)
        return S, L

    def compute_clustering_energy(self, C, L, S):
        """
        compute fair clustering energy
        """
        e_dist = self.compute_a_p(L, C)
        clustering_E = (S * e_dist).sum()
        clustering_E_discrete = [km_discrete_energy(e_dist, L, k) for k in range(self.K)]
        clustering_E_discrete = sum(clustering_E_discrete)
        return clustering_E, clustering_E_discrete


class KmeansUtils(KUtils):
    def compute_a_p(self, L, C):  # noqa: ARG002
        a_p = ecdist(self.X, C, squared=True)
        return a_p

    def step(self, L, C):
        tmp_list = [np.where(k == L)[0] for k in range(self.K)]
        C_list = [self.kmeans_update(t) for t in tmp_list]
        C = np.asarray(np.vstack(C_list))
        sqdist = ecdist(self.X, C, squared=True)
        a_p = sqdist.copy()
        return a_p, C

    def kmeans_update(self, tmp):
        """ """
        X_tmp = self.X[tmp, :]
        c1 = X_tmp.mean(axis=0)

        return c1


class KmedianUtils(KUtils):
    def compute_a_p(self, L, C):  # noqa: ARG002
        a_p = ecdist(self.X, C)
        return a_p

    def step(self, L, C):
        tmp_list = [np.where(k == L)[0] for k in range(self.K)]
        C_list = [self.kmedian_update(x) for x in tmp_list]
        C = np.asarray(np.vstack(C_list))
        sqdist = ecdist(self.X, C)
        a_p = sqdist.copy()
        return a_p, C

    def kmedian_update(self, tmp):
        # print("ID of process running worker: {}".format(os.getpid()))
        X_tmp = self.X[tmp, :]
        D = pdist_chunk(X_tmp, reduce_func=reduce_func)
        J = next(D)
        j = np.argmin(J)
        c1 = X_tmp[j, :]
        return c1


class NCutUtils:
    def __init__(self, X, K):
        self.K = K
        self.N = len(X)
        self.X = X
        self.knn = 20

    def compute_a_p(self, L, C):
        self.A = create_affinity(self.X, self.knn)
        self.d = self.A.sum(axis=1)
        S = get_S_discrete(L, self.N, self.K)
        sqdist_list = [KernelBound_k(self.A, self.d, S[:, k], self.N) for k in range(self.K)]
        sqdist = np.asarray(np.vstack(sqdist_list).T)
        a_p = sqdist.copy()
        return a_p, C

    def step(self, L, C):  # noqa: ARG002
        S = get_S_discrete(L, self.N, self.K)
        sqdist_list = [KernelBound_k(self.A, self.d, S[:, k], self.N) for k in range(self.K)]
        sqdist = np.asarray(np.vstack(sqdist_list).T)
        a_p = sqdist.copy()
        return a_p

    def update(self, a_p, L, C):  # noqa: ARG002
        L = a_p.argmin(axis=1)
        S = get_S_discrete(L, self.N, self.K)
        return S, L

    def compute_clustering_energy(self, C, L, S):  # noqa: ARG002
        """
        compute fair clustering energy
        """
        clustering_E = NormalizedCutEnergy(self.A, S, L)
        clustering_E_discrete = NormalizedCutEnergy_discrete(self.A, L)
        return clustering_E, clustering_E_discrete
