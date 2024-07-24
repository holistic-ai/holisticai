Variational Fair Clustering
-----------------

.. note::
    **Learning tasks:** Clustering.

Introduction
~~~~~~~~~~~~~~~~
The Variational Fair Clustering method integrates a Kullback-Leibler (KL) fairness term with a variety of clustering objectives, including prototype-based and graph-based methods. This approach allows for controlling the trade-off between fairness and clustering quality, providing a scalable solution with convergence guarantees. Unlike existing combinatorial and spectral methods, this variational framework does not require eigenvalue decomposition, making it suitable for large-scale datasets.

Description
~~~~~~~~~~~~~~~~
The problem involves assigning :math:`N` data points to :math:`K` clusters while ensuring that the clusters are balanced with respect to :math:`J` different demographic groups. The method introduces a fairness term based on the Kullback-Leibler (KL) divergence to measure the deviation from a target demographic distribution within each cluster.

- **Problem Definition:**
  Given a set of data points :math:`X = \{ x_p \in \mathbb{R}^M, p = 1, \ldots, N \}` and :math:`J` demographic groups, the goal is to assign these points to :math:`K` clusters such that the clusters are fair with respect to the demographic groups.

- **Main Characteristics:**
  - Integrates a KL fairness term with clustering objectives.
  - Provides a tight upper bound based on concave-convex decomposition.
  - Scalable and suitable for large datasets.
  - Allows independent updates for each assignment variable, facilitating distributed computation.

- **Step-by-Step Description:**

  1. **Initialization:**
  
     - Initialize cluster labels from initial seeds.
     - Initialize the soft cluster-assignment vector :math:`S` from the labels.
  2. **Iterative Optimization:**

     - Compute the auxiliary variable :math:`a_p` from :math:`S`.
     - Initialize the assignment probabilities :math:`s_p`.
     - Update :math:`s_p` using the derived bound optimization.
     - Repeat until the objective function converges.
  3. **Final Assignment:**

     - Assign each data point to the cluster with the highest probability.

References
~~~~~~~~~~~~~~~~
1. Ziko, Imtiaz Masud, et al. "Variational fair clustering." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.