import numpy as np
from holisticai.bias.mitigation.postprocessing.mcmf_clustering.utils.algorithm_utils import Utils


class OPTModification:
    def __init__(self, k, verbose=0):
        self.k = k
        self.verbose = verbose

    def fit(self, y_pred, p_attr):
        y_pred = self.run(y_pred, p_attr)

    def run(self, y_pred, p_attr):
        Nx, q, r, lower_bound, upper_bound, Cj = self.init_parameters(y_pred, p_attr)
        self.utils = Utils(upper_bound=upper_bound, lower_bound=lower_bound, Nx=Nx, k=self.k)

        if r == 0:
            gt, gp = [], []
            for j in range(self.k):
                if Cj[j]["beta"] > Nx // self.k:
                    gt.append(j)
                if Cj[j]["beta"] <= Nx // self.k:
                    gp.append(j)
            T, Cj = self.utils.select_items(gt, Cj, mode="exact")
            T, Cj = self.utils.choose_items(gp, Cj, T, mode="exact")
        else:
            g1, g2, g3, g4 = [], [], [], []
            for j in range(self.k):
                if Cj[j]["beta"] > upper_bound:
                    g1.append(j)
                if Cj[j]["beta"] == upper_bound:
                    g2.append(j)
                if Cj[j]["beta"] == lower_bound:
                    g3.append(j)
                if Cj[j]["beta"] < lower_bound:
                    g4.append(j)

            t = len(g1)
            p = len(g1 + g2)
            # m = len(g1 + g2 + g3)

            gs = g1 + g2 + g3 + g4

            if p < r:
                T, Cj = self.utils.select_items(g1, Cj, mode="upper")
                rp = r - p
                g = g3 + g4
                T, Cj = self.utils.choose_items(g[:rp], Cj, T, mode="upper")
                T, Cj = self.utils.choose_items(g[rp:], Cj, T, mode="lower")
            elif p == r:
                T, Cj = self.utils.select_items(g1, Cj, mode="upper")
                T, Cj = self.utils.choose_items(g4, Cj, T, mode="lower")
            elif p > r:
                if t > r:
                    T, Cj = self.utils.select_items(g1, Cj, mode="upper")
                    T, Cj = self.utils.select_items(gs[r:p], Cj, mode="lower")
                    T, Cj = self.utils.choose_items(g4, Cj, T, mode="lower")

                elif t == r:
                    T, Cj = self.utils.select_items(g1, Cj, mode="upper")
                    T, Cj = self.utils.select_items(g2, Cj, mode="lower")
                    T, Cj = self.utils.choose_items(g4, Cj, T, mode="lower")

                elif t < r:
                    T, Cj = self.utils.select_items(g1, Cj, mode="upper")
                    start = r - t
                    end = p
                    T, Cj = self.utils.select_items(gs[start:end], Cj, mode="lower")
                    T, Cj = self.utils.choose_items(g4, Cj, T, mode="lower")

        for js in range(self.k):
            jr = Cj[js]["jr"]
            index = Cj[js]["index"]
            y_pred[index] = jr

        return y_pred, T

    def init_parameters(self, y_pred, p_attr):
        Nx = np.sum(p_attr == 1)
        q = Nx // self.k
        r = Nx - q * self.k

        if r >= 1:
            lower_bound = q
            upper_bound = q + 1
        else:
            lower_bound = upper_bound = q

        cluster_info = []
        for j in range(self.k):
            c_index = np.where((y_pred == j) & (p_attr == 1))[0]
            beta_j = len(c_index)
            cluster_info.append((j, beta_j, c_index))

        Cj = {}
        for js, (jr, beta, index) in enumerate(sorted(cluster_info, key=lambda x: x[1], reverse=True)):
            Cj[js] = {"jr": jr, "beta": beta, "index": list(index)}

        return Nx, q, r, lower_bound, upper_bound, Cj
