import random

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm


class KMediamClusteringAlgorithm:
    def __init__(
        self,
        n_clusters=2,
        init_centers="Random",
        max_iter=1000,
        strategy="LS",
        verbose=False,
    ):
        self.k = n_clusters
        self.init_centers = init_centers
        self.max_iter = max_iter
        self.verbose = verbose
        self.strategy = strategy

    def _compute_new_assigment_and_cost(self, centers):
        cost = self._compute_cost(centers)
        assignment = self._compute_new_assigment(centers)
        return assignment, cost

    def _compute_cost(self, centers):
        centers = np.array(centers, dtype=np.int32)
        group_costs = []
        for gid in self.group_ids:
            group_costs.append(
                np.mean(
                    np.amin(self.distances[np.ix_(self.p_attr == gid, centers)], axis=1)
                )
            )
        cost = np.max(group_costs)
        return cost

    def _compute_new_assigment(self, centers):
        centers = np.array(centers, dtype=np.int32)
        assignment = centers[
            np.argmin(self.distances[np.ix_(np.arange(self.n), centers)], axis=1)
        ]
        return assignment

    def fit(self, X, p_attr):
        self.n = len(X)
        self.group_ids = np.unique(p_attr)
        distances = pairwise_distances(X, Y=X, metric="l1")

        self.distances = distances
        self.p_attr = p_attr

        if self.strategy == "LS":
            self._linear_search(X)

        elif self.strategy == "GA":
            self._genetic_algorithm(X)

    def _linear_search(self, X):

        if self.init_centers == "KMedoids":
            from holisticai.utils.models.cluster import KMedoids

            kmedoids = KMedoids(n_clusters=self.k).fit(X)
            starting_centers = kmedoids.medoid_indices_

        elif self.init_centers == "Random":
            starting_centers = random.sample(range(1, self.n - 1), self.k)

        if starting_centers[1] > starting_centers[0] - 1:
            starting_centers[1] += 1

        chosen_centers = starting_centers
        centers = starting_centers

        assignment, min_cost = self._compute_new_assigment_and_cost(centers)

        flag = 0
        for _ in range(self.max_iter):
            flag = 1
            # Check if any other point is a better center
            r = list(range(self.n))
            random.shuffle(r)
            t = tqdm(r, desc=f"Cost: {min_cost:.4f}", leave=True)
            for c in t:
                for i in range(self.k):
                    centers = list(chosen_centers)
                    # Try replacing i center
                    if not (c in centers):
                        # set new center
                        centers[i] = c
                        # do new assignment calculate new cost
                        assignment, curr_cost = self._compute_new_assigment_and_cost(
                            centers
                        )
                        # change variables if new cost is better
                        if curr_cost < min_cost:
                            min_cost = curr_cost
                            t.set_description(f"Cost: {min_cost:.4f}")
                            t.refresh()  # to show immediately the update
                            chosen_centers = centers
                            chosen_assignment = assignment
                            flag = 0
                            break

            if flag == 1:
                break

        self.labels_ = np.array(chosen_assignment)
        self.cluster_centers_ = X[np.array(chosen_centers)]
        self.centers = np.array(chosen_centers)

    def _genetic_algorithm(self, X):
        optimization_function = lambda x: self._compute_cost(x)

        from .....utils.optimizers import GAHiperparameters, GeneticAlgorithm

        varbound = np.array([[0, len(X) - 1]] * self.k)

        ga_kargs = {
            "max_num_iteration": self.max_iter,
            "population_size": 100,
            "mutation_probability": 0.1,
            "elit_ratio": 0.01,
            "crossover_probability": 0.5,
            "parents_portion": 0.3,
            "crossover_type": "uniform",
            "max_iteration_without_improv": None,
        }

        algorithm_parameters = GAHiperparameters(**ga_kargs)

        optimizer = GeneticAlgorithm(
            function=optimization_function,
            dimension=self.k,
            variable_type="int",
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_parameters,
            verbose=self.verbose,
        )

        optimizer.run()

        chosen_centers = np.array(optimizer.best_variable, dtype=np.int32)
        chosen_assignment = self._compute_new_assigment(chosen_centers)

        self.labels_ = np.array(chosen_assignment)
        self.cluster_centers_ = X[chosen_centers]
        self.centers = chosen_centers
