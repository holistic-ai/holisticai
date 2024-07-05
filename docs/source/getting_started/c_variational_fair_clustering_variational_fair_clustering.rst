**Variational Fair Clustering**
===============================

**Introduction**
----------------
The Variational Fair Clustering method integrates a Kullback-Leibler (KL) fairness term with a variety of clustering objectives, including prototype-based and graph-based methods. This approach allows for controlling the trade-off between fairness and clustering quality, providing a scalable solution with convergence guarantees. Unlike existing combinatorial and spectral methods, this variational framework does not require eigenvalue decomposition, making it suitable for large-scale datasets.

**Description**
---------------
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

**Equations/Algorithms**
------------------------
The fairness measure is defined as:

.. math::
    \text{balance}(S_k) = \min_{j \neq j'} \frac{V_j^T S_k}{V_{j'}^T S_k} \in [0, 1]

The overall clustering balance is the minimum of the balance measures across all clusters:

.. math::
    \text{balance} = \min_k \text{balance}(S_k)

The KL fairness term is given by:

.. math::
    D_{KL}(U \| P_k) = \sum_{j=1}^J U_j \log \left( \frac{U_j}{P_{k,j}} \right)

The proposed algorithm can be summarized in the following pseudocode:

.. math::
    :label: algorithm

    \begin{aligned}
    &\text{Input: } X, \text{Initial seeds}, \lambda, U, \{V_j\}_{j=1}^J \\
    &\text{Output: Clustering labels} \in \{1, \ldots, K\}^N \\
    &\text{Initialize labels from initial seeds.} \\
    &\text{Initialize } S \text{ from labels.} \\
    &\text{Initialize } i = 1. \\
    &\text{repeat} \\
    &\quad \text{Compute } a_p^i \text{ from } S \\
    &\quad \text{Initialize } s_p^i = \exp(-a_p^i) / 1^t \exp(-a_p^i). \\
    &\quad \text{repeat} \\
    &\quad \quad \text{Compute } s_p^{i+1} \text{ using (13).} \\
    &\quad \quad s_p^i \leftarrow s_p^{i+1}. \\
    &\quad \quad S = [s_p^i]; \forall p. \\
    &\quad \text{until } A_i(S) \text{ in (11) does not change} \\
    &\quad i = i + 1. \\
    &\text{until } E(S) \text{ in (4) does not change} \\
    &l_p = \arg \max_k s_{p,k}; \forall p. \\
    &\text{labels} = \{l_p\}_{p=1}^N.
    \end{aligned}

**Usage Examples**
------------------
The method was tested on various datasets, including synthetic datasets and real-world datasets such as Adult, Bank, and Census II. It was shown to achieve competitive fairness and clustering objectives across these datasets.

- **Synthetic Datasets:** Created with different demographic proportions to test the balance and fairness of the clustering.
- **Adult Dataset:** Contains demographic information used to evaluate the fairness of clustering in a real-world scenario.
- **Bank Dataset:** Used to test the method's scalability and effectiveness with multiple demographic groups.
- **Census II Dataset:** Demonstrates the method's ability to handle large-scale data.

**Advantages and Limitations**
------------------------------
*Advantages:*

- Allows control over the trade-off between fairness and clustering quality.
- Scalable to large datasets due to independent updates for each assignment variable.
- Does not require eigenvalue decomposition, reducing computational complexity.
- Applicable to both prototype-based and graph-based clustering objectives.

*Limitations:*

- The method's performance may depend on the choice of the initial seeds.
- The fairness term may dominate the clustering objective for small values of the trade-off parameter :math:`\lambda`.
- Requires careful tuning of the trade-off parameter :math:`\lambda` to balance fairness and clustering quality.

**References**
---------------
1. Ziko, Imtiaz Masud, et al. "Variational fair clustering." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.