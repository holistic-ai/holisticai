import logging
import math
from collections import defaultdict

import numpy as np
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._bound_update import (
    BoundUpdate,
    normalize_2,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._logger import MFLogger
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._method_utils import (
    _init_centroids,
    compute_fairness_energy,
    ecdist,
    km_le,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._methods import (
    KmeansUtils,
    KmedianUtils,
    NCutUtils,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._utils import (
    get_fair_accuracy_proportional,
)
from sklearn.cluster import KMeans

logger = logging.getLogger()


class Monitor:
    def __init__(self):
        self.history = defaultdict(list)

    def save(self, **kargs):
        for k, v in kargs.items():
            self.history[k].append(v)


def km_init(K, X, C_init, l_init=None, random_state=np.random.RandomState()):  # noqa: B008
    """
    Initial seeds
    """
    if isinstance(C_init, str):
        if C_init == "kmeans_plus":
            M = _init_centroids(X, n_clusters=K, init="k-means++", random_state=random_state)
            L = km_le(X, M)

        elif C_init == "kmeans":
            kmeans = KMeans(n_clusters=K).fit(X)
            L = kmeans.labels_
            M = kmeans.cluster_centers_
    else:
        M = C_init.copy()
        # l = km_le(X,M)
        L = l_init.copy()

    del C_init, l_init

    return M, L


class FairnessUtility:
    def restore_nonempty_cluster(self, X, K, oldl, oldC, oldS):  # noqa: ARG002
        ts_limit = 2
        C_init = "kmeans"
        if self.ts > ts_limit:
            logger.info("not having some labels")
            trivial_status = True
            L = oldl.copy()
            C = oldC.copy()
            S = oldS.copy()

        else:
            logger.info("try with new seeds")

            C, L = km_init(self.K, X, C_init)
            sqdist = ecdist(X, C, squared=True)
            S = normalize_2(np.exp(-sqdist))
            trivial_status = False

        return L, C, S, trivial_status

    def fit_fairness(self, X, a_p, group_prob, groups_ids):
        trivial_status = False
        l_check = a_p.argmin(axis=1)

        # Check for empty cluster
        if len(np.unique(l_check)) != self.K:
            L, C, S, trivial_status = self.restore_nonempty_cluster(
                X, self.K, self.old["l"], self.old["C"], self.old["S"]
            )
            self.ts = self.ts + 1
            return S, L, trivial_status

        L, S, bound_E = self.bound_update.transform(a_p, group_prob, groups_ids)
        return S, L, trivial_status


METHODS_CATALOG = {"kmeans": KmeansUtils, "kmedian": KmedianUtils, "ncut": NCutUtils}


class FairClustering(FairnessUtility):
    def __init__(self, K, L, lmbda, method, max_iter=100, verbose=True):
        self.K = K
        self.lmbda = lmbda
        self.L = L
        self.bound_update = BoundUpdate(bound_lambda=lmbda, L=L, bound_iteration=10000)
        self.method = method
        self.monitor = Monitor()
        self.max_iter = max_iter
        self.logger = MFLogger(total_iterations=self.max_iter, verbose=verbose)

    def fit(self, X, group_prob, groups_ids, random_state):
        C, L = km_init(self.K, X, "kmeans_plus", random_state=random_state)

        self.clustering_method = METHODS_CATALOG[self.method](X, self.K)
        self.ts = 0
        S = []
        fairness_error = 0.0
        oldE = 1e100
        for i in range(self.max_iter):
            self.old = {"C": C.copy(), "l": L.copy(), "S": S.copy()}

            if i == 0:
                a_p = self.clustering_method.compute_a_p(L, C)
            else:
                a_p, C = self.clustering_method.step(L, C)

            S, L, trivial_status = self.fit_fairness(X, a_p, group_prob, groups_ids)
            if trivial_status:
                break

            (
                clusterE,
                clusterE_discrete,
            ) = self.clustering_method.compute_clustering_energy(C, L, S)
            fairness_error = get_fair_accuracy_proportional(group_prob, groups_ids, L, self.K)
            fairE = compute_fairness_energy(group_prob, groups_ids, S, self.lmbda)
            currentE = clusterE + fairE

            self.monitor.save(
                fair_cluster_E=currentE,
                cluster_E=clusterE,
                fair_E=fairE,
                cluster_E_discrete=clusterE_discrete,
                fairness_error=fairness_error,
            )

            self.logger.update(
                iteration=i,
                fairness_error=fairness_error,
                fair_cluster_energy=currentE,
                cluster_energy=clusterE_discrete,
            )

            if (len(np.unique(L)) != self.K) or math.isnan(fairness_error):
                L, C, S, trivial_status = self.restore_nonempty_cluster(
                    X, self.K, self.old["l"], self.old["C"], self.old["S"]
                )
                self.ts = self.ts + 1
                if trivial_status:
                    break

            if i > 1 and (abs(currentE - oldE) <= 1e-4 * abs(oldE)):
                break

            oldE = currentE.copy()

        self.C = C
        self.l = L
        self.S = S
        self.group_prob = group_prob
        return self

    def predict(self, X, groups_ids):
        S = ecdist(X, self.C, squared=True)
        L = np.argmax(S, axis=1)
        C = self.C
        N, D = X.shape
        group_prob = self.group_prob

        self.clustering_method = METHODS_CATALOG[self.method](X, self.K)
        a_p = self.clustering_method.compute_a_p(L, C)
        _, L, _ = self.fit_fairness(X, a_p, group_prob, groups_ids)
        return L
