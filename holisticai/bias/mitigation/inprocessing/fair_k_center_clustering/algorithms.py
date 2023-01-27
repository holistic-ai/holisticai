import itertools

import numpy as np
import scipy.sparse.csgraph


def fair_k_center_exact(dmat, p_attr, nr_centers_per_group, given_centers):
    """
    Description
    -----------
    Exhaustive search to exactly solve the fair k-center problem
    Obs: only works for small problem instances.

    Parameters
    ----------
    dmat : matrix-like
        distance matrix of size nxn
    p_attr :  array-like
        integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_group: array-like
        integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers : array-like
        integer-vector with entries in 0,...,n-1

    Return
    -------
    (optimal centers, clustering, optimal fair k-center cost)
    """

    n = dmat.shape[0]
    m = nr_centers_per_group.size
    k = np.sum(nr_centers_per_group)

    cost = np.inf
    best_choice = []

    for mmm in itertools.combinations(np.arange(n), k):

        cluster_centers = np.array(mmm)

        curr_nr_clusters_per_sex = np.zeros(m)
        for ell in np.arange(m):
            curr_nr_clusters_per_sex[ell] = np.sum(p_attr[cluster_centers] == ell)

        if sum(curr_nr_clusters_per_sex == nr_centers_per_group) == m:
            curr_cost = np.amax(
                np.amin(
                    dmat[
                        np.ix_(
                            np.hstack((cluster_centers, given_centers)), np.arange(n)
                        )
                    ],
                    axis=0,
                )
            )
        else:
            curr_cost = np.inf

        if curr_cost < cost:
            cost = curr_cost
            best_choice = cluster_centers.copy()

    clustering = np.array(
        [
            np.argmin(dmat[ell, np.hstack((best_choice, given_centers))])
            for ell in np.arange(n)
        ]
    )

    return best_choice, clustering, cost


def k_center_greedy_with_given_centers(dmat, k, given_centers):
    """
    Description
    -----------
        Implementation of Algorithm 1.

    Parameters
    ----------
    dmat : matrix-like
        distance matrix of size nxn
    k : int
        integer smaller than n
    given_centers : array-like
        integer-vector with entries in 0,...,n-1

    Return
    ------
        approx. optimal centers
    """

    n = dmat.shape[0]

    if k == 0:
        cluster_centers = np.array([], dtype=int)
    else:
        if given_centers.size == 0:
            cluster_centers = np.random.choice(n, 1, replace=False)
            kk = 1
        else:
            cluster_centers = given_centers
            kk = 0

        distance_to_closest = np.amin(
            dmat[np.ix_(cluster_centers, np.arange(n))], axis=0
        )
        while kk < k:
            temp = np.argmax(distance_to_closest)
            cluster_centers = np.append(cluster_centers, temp)
            distance_to_closest = np.amin(
                np.vstack((distance_to_closest, dmat[temp, :])), axis=0
            )
            kk += 1

        cluster_centers = cluster_centers[given_centers.size :]

    return cluster_centers


def fair_k_center_APPROX(dmat, p_attr, nr_centers_per_group, given_centers):
    """
    Description
    -----------
        Implementation of Algorithm 4.

    Parameters
    ----------
    dmat : array-like
        distance matrix of size nxn
    p_attr :  array-like
        integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_group : array-like
        integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers : array-like
        integer-vector with entries in 0,...,n-1

    Return
    ------
        approx. optimal centers
    """

    n = dmat.shape[0]
    m = nr_centers_per_group.size
    k = np.sum(nr_centers_per_group)

    cluster_centersTE = k_center_greedy_with_given_centers(dmat, k, given_centers)

    CURRENT_nr_clusters_per_sex = np.zeros(m, dtype=int)
    for ell in np.arange(k):
        CURRENT_nr_clusters_per_sex[p_attr[cluster_centersTE[ell]]] += 1

    partition = np.array(
        [
            np.argmin(dmat[ell, np.hstack((cluster_centersTE, given_centers))])
            for ell in np.arange(n)
        ]
    )
    selection = partition < k
    selected_partition = partition[selection]
    selected_p_attr = p_attr[selection]
    selected_index = np.array(
        [
            np.where(np.arange(n)[selection] == cluster_centersTE[ell])[0][0]
            for ell in np.arange(k)
        ]
    )
    G, centersTE = swapping_graph(
        selected_partition, selected_index, selected_p_attr, nr_centers_per_group
    )
    cluster_centersTE = np.arange(n)[partition < k][centersTE]

    if G.size == 0:
        cluster_centers = cluster_centersTE
    else:
        new_data_set = np.array([], dtype=int)
        new_given_centersT = np.array([], dtype=int)
        for ell in np.arange(k):
            if np.isin(p_attr[cluster_centersTE[ell]], G):
                new_data_set = np.hstack((new_data_set, np.where(partition == ell)[0]))
            else:
                new_given_centersT = np.hstack(
                    (new_given_centersT, cluster_centersTE[ell])
                )
        new_given_centers = np.hstack((new_given_centersT, given_centers))
        p_attr_new = p_attr[new_data_set]
        p_attr_newT = np.zeros(new_data_set.size, dtype=int)
        cc = 0
        for ell in G:
            p_attr_newT[p_attr_new == ell] = cc
            cc += 1
        new_data_set = np.hstack((new_data_set, new_given_centers))
        p_attr_newT = np.hstack(
            (p_attr_newT, np.zeros(new_given_centers.size, dtype=int))
        )

        cluster_centers_rek = fair_k_center_APPROX(
            dmat[np.ix_(new_data_set, new_data_set)],
            p_attr_newT,
            nr_centers_per_group[G],
            np.arange(new_data_set.size - new_given_centers.size, new_data_set.size),
        )

        new_given_centersT_additional = np.array([], dtype=int)
        for ell in np.setdiff1d(np.arange(m), G):
            if np.sum(p_attr[new_given_centersT] == ell) < nr_centers_per_group[ell]:
                toadd = nr_centers_per_group[ell] - np.sum(
                    p_attr[new_given_centersT] == ell
                )
                toadd_pot = np.setdiff1d(np.where(p_attr == ell)[0], new_given_centersT)
                if toadd_pot.size > toadd:
                    new_given_centersT_additional = np.hstack(
                        (new_given_centersT_additional, toadd_pot[0:toadd])
                    )
                else:
                    new_given_centersT_additional = np.hstack(
                        (new_given_centersT_additional, toadd_pot)
                    )

        cluster_centers = np.hstack(
            (
                new_given_centersT,
                new_given_centersT_additional,
                new_data_set[cluster_centers_rek],
            )
        )

    return cluster_centers


def swapping_graph(partition, centers, p_attr, nr_centers_per_group):
    """
    Description
    -----------
        Implementation of Algorithm 3.

    Parameters
    ----------
        partition : array-like
            integer-vector of length n with entries in 0 ... k-1
        centers : array-like
            integer-vector of length k with entries in 0 ... n-1
        p_attr : array-like
            integer-vector of length n with entries in 0 ... m-1
        nr_centers_per_group : array-like
            integer-vector of length m with entries in 0,...,k and sum over entries equaling k

    Return
    ------
        (G, swapped centers)
    """

    n = partition.size
    m = nr_centers_per_group.size
    k = centers.size

    CURRENT_nr_clusters_per_sex = np.zeros(m, dtype=int)
    for ell in np.arange(k):
        CURRENT_nr_clusters_per_sex[p_attr[centers[ell]]] += 1

    sex_of_assigned_center = p_attr[centers[partition]]
    Adja = np.zeros((m, m))
    for ell in np.arange(n):
        Adja[sex_of_assigned_center[ell], p_attr[ell]] = 1

    dmat_gr, predec = scipy.sparse.csgraph.shortest_path(
        Adja, directed=True, return_predecessors=True
    )

    is_there_a_path = 0
    for ell in np.arange(m):
        for zzz in np.arange(m):
            if (CURRENT_nr_clusters_per_sex[ell] > nr_centers_per_group[ell]) and (
                CURRENT_nr_clusters_per_sex[zzz] < nr_centers_per_group[zzz]
            ):
                if dmat_gr[ell, zzz] != np.inf:
                    path = np.array([zzz])
                    while path[0] != ell:
                        path = np.hstack((predec[ell, path[0]], path))
                    is_there_a_path = 1
                    break
        if is_there_a_path == 1:
            break

    while is_there_a_path:

        for hhh in np.arange(path.size - 1):
            for ell in np.arange(n):
                if (p_attr[ell] == path[hhh + 1]) and (
                    sex_of_assigned_center[ell] == path[hhh]
                ):
                    centers[partition[ell]] = ell
                    sex_of_assigned_center[partition == partition[ell]] = p_attr[ell]
                    break
        CURRENT_nr_clusters_per_sex[path[0]] -= 1
        CURRENT_nr_clusters_per_sex[path[-1]] += 1

        Adja = np.zeros((m, m))
        for ell in np.arange(n):
            Adja[sex_of_assigned_center[ell], p_attr[ell]] = 1

        dmat_gr, predec = scipy.sparse.csgraph.shortest_path(
            Adja, directed=True, return_predecessors=True
        )

        is_there_a_path = 0
        for ell in np.arange(m):
            for zzz in np.arange(m):
                if (CURRENT_nr_clusters_per_sex[ell] > nr_centers_per_group[ell]) and (
                    CURRENT_nr_clusters_per_sex[zzz] < nr_centers_per_group[zzz]
                ):
                    if dmat_gr[ell, zzz] != np.inf:
                        path = np.array([zzz])
                        while path[0] != ell:
                            path = np.hstack((predec[ell, path[0]], path))
                        is_there_a_path = 1
                        break
            if is_there_a_path == 1:
                break

    if sum(CURRENT_nr_clusters_per_sex == nr_centers_per_group) == m:
        return np.array([]), centers
    else:

        G = np.where(CURRENT_nr_clusters_per_sex > nr_centers_per_group)[0]
        for ell in np.arange(m):
            for zzz in np.arange(m):
                if ((dmat_gr[ell, zzz] != np.inf) and np.isin(ell, G)) and (
                    not np.isin(zzz, G)
                ):
                    G = np.hstack((G, zzz))

        return G, centers


def heuristic_greedy_on_each_group(dmat, p_attr, nr_centers_per_group, given_centers):
    """
    Description
    -----------
        Implementation of Heuristic A as described in Section 5.3.

    Parameters
    ----------
    dmat : matrix-like
        distance matrix of size nxn
    p_attr : array-like
        integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_group : array-like
        integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers : array-like
        integer-vector with entries in 0,...,n-1

    Return
    ------
        heuristically chosen centers
    """

    m = nr_centers_per_group.size

    cluster_centers = np.array([], dtype=int)

    for ell in np.arange(m):
        subgroup = np.where(p_attr == ell)[0]
        given_centers_subgroup = np.where(np.isin(subgroup, given_centers))[0]

        cent_subgroup = k_center_greedy_with_given_centers(
            dmat[np.ix_(subgroup, subgroup)],
            nr_centers_per_group[ell],
            given_centers_subgroup,
        )

        cluster_centers = np.hstack((cluster_centers, subgroup[cent_subgroup]))

    return cluster_centers


def heuristic_greedy_till_constraint_is_satisfied(
    dmat, p_attr, nr_centers_per_group, given_centers
):
    """
    Description
    -----------
        Implementation of Heuristic B as described in Section 5.3.

    Parameters
    ----------
    dmat : matrix-like
        distance matrix of size nxn
    p_attr : array-like
        integer-vector of length n with entries in 0,...,m-1, where m is the number of groups
    nr_centers_per_group : array-like
        integer-vector of length m with entries in 0,...,k and sum over entries equaling k
    given_centers : array-like
        integer-vector with entries in 0,...,n-1

    Return
    ------
        heuristically chosen centers

    """

    n = dmat.shape[0]
    m = nr_centers_per_group.size
    k = np.sum(nr_centers_per_group)

    current_nr_per_sex = np.zeros(m)

    if k == 0:
        cluster_centers = np.array([], dtype=int)
    else:
        if given_centers.size == 0:
            cluster_centers = np.random.choice(n, 1, replace=False)
            current_nr_per_sex[p_attr[cluster_centers]] += 1
            kk = 1
        else:
            cluster_centers = given_centers
            kk = 0

        distance_to_closest = np.amin(
            dmat[np.ix_(cluster_centers, np.arange(n))], axis=0
        )
        while kk < k:
            feasible_groups = np.where(current_nr_per_sex < nr_centers_per_group)[0]
            feasible_points = np.where(np.isin(p_attr, feasible_groups))[0]
            new_point = feasible_points[np.argmax(distance_to_closest[feasible_points])]
            current_nr_per_sex[p_attr[new_point]] += 1
            cluster_centers = np.append(cluster_centers, new_point)
            distance_to_closest = np.amin(
                np.vstack((distance_to_closest, dmat[new_point, :])), axis=0
            )
            kk += 1

        cluster_centers = cluster_centers[given_centers.size :]

    return cluster_centers
