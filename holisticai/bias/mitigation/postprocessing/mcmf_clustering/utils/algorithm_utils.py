import random


class Utils:
    def __init__(self, upper_bound, lower_bound, Nx, k):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.Nx = Nx
        self.k = k

    def select_items(self, group_cluster, Cj, mode="upper"):
        T = []
        for js in group_cluster:
            beta = Cj[js]["beta"]
            if mode == "upper":
                threshold = self.upper_bound
            elif mode == "exact":
                threshold = self.Nx // self.k
            else:
                threshold = self.lower_bound

            n_items_to_container = beta - threshold

            index = Cj[js]["index"]
            random.shuffle(index)
            index_selected, index = (
                index[:n_items_to_container],
                index[n_items_to_container:],
            )

            assert len(index) == threshold

            Cj[js]["index"] = index
            Cj[js]["beta"] = len(index)
            T += index_selected
        return T, Cj

    def choose_items(self, group_cluster, Cj, T, mode="upper"):
        for js in group_cluster:
            beta = Cj[js]["beta"]
            if mode == "upper":
                threshold = self.upper_bound
            elif mode == "exact":
                threshold = self.Nx // self.k
            else:
                threshold = self.lower_bound

            n_items_from_container = threshold - beta
            assert n_items_from_container >= 0
            selected_index, T = T[:n_items_from_container], T[n_items_from_container:]
            Cj[js]["index"] += selected_index
            Cj[js]["beta"] = len(Cj[js]["index"])
        return T, Cj
