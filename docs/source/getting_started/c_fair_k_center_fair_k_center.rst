**Fair k-Center Clustering Method**
=================

**Introduction**
----------------
The Fair k-Center Clustering method addresses the problem of centroid-based clustering, such as k-center, in a way that ensures fair representation of different demographic groups. This method is particularly relevant in scenarios where the data set comprises multiple demographic groups, and there is a need to select a fixed number of representatives (centers) from each group to form a summary. The method extends the traditional k-center clustering problem by incorporating fairness constraints, ensuring that each group is fairly represented in the selected centers.

**Description**
---------------
The Fair k-Center Clustering method aims to minimize the maximum distance between any data point and its closest center while ensuring that a specified number of centers are chosen from each demographic group. The problem can be formally defined as follows:

Given:

- A finite data set :math:`S`
- A metric :math:`d: S \times S \to \mathbb{R}_{\geq 0}` satisfying the triangle inequality
- A partition of :math:`S` into :math:`m` demographic groups :math:`S = \bigcup_{i=1}^m S_i`
- A parameter :math:`k \in \mathbb{N}` representing the total number of centers
- Parameters :math:`k_{S_i} \in \mathbb{N}_0` for each group :math:`S_i` such that :math:`\sum_{i=1}^m k_{S_i} = k`
- An optional subset :math:`C_0 \subseteq S` of pre-specified centers

The objective is to find a set of centers :math:`C = \{c_1, \ldots, c_k\} \subseteq S` such that:
- :math:`|C \cap S_i| = k_{S_i}` for all :math:`i = 1, \ldots, m`
- The maximum distance from any point in :math:`S` to its closest center in :math:`C \cup C_0` is minimized

The method involves a recursive algorithm that handles the fairness constraints by iteratively selecting centers and ensuring the required representation from each group. The algorithm can be broken down into the following steps:

1. **Initialization**: Start with an empty set of centers and the given parameters.
2. **Center Selection**: Use a greedy strategy to select centers that maximize the distance to the current set of centers.
3. **Fairness Adjustment**: Adjust the selected centers to ensure the required number of centers from each group.
4. **Recursion**: If the fairness constraints are not met, recursively apply the algorithm to a subset of the data until the constraints are satisfied.

**Equations/Algorithms**
------------------------

The main algorithm for the Fair k-Center Clustering method is presented below:

.. math::
    :label: algorithm-4

    \begin{aligned}
    &\text{Input: metric } d: S \times S \to \mathbb{R}_{\geq 0}; k_{S_1}, \ldots, k_{S_m} \in \mathbb{N}_0 \text{ with } \sum_{i=1}^m k_{S_i} = k; C_0 \subseteq S; \\
    &\text{group-membership vector } \in \{1, \ldots, m\}^{|S|} \\
    &\text{Output: } C_A = \{c_1^A, \ldots, c_k^A\} \subseteq S \\
    &\text{1: run Algorithm 1 on } S \text{ with } k = \sum_{i=1}^m k_{S_i} \text{ and } C_0' = C_0; \text{ let } \tilde{C}_A = \{\tilde{c}_1^A, \ldots, \tilde{c}_k^A\} \text{ denote its output} \\
    &\text{2: if } m = 1 \\
    &\quad \text{return } \tilde{C}_A \\
    &\text{3: form clusters } L_1, \ldots, L_k, L_1', \ldots, L_{|C_0|}' \text{ by assigning every } s \in S \text{ to its closest center in } \tilde{C}_A \cup C_0 \\
    &\text{4: apply Algorithm 3 to } \tilde{c}_1^A, \ldots, \tilde{c}_k^A \text{ and } \bigcup_{i=1}^k L_i \text{ in order to exchange some centers } \tilde{c}_i^A \text{ and obtain } G \subseteq \{S_1, \ldots, S_m\} \\
    &\text{5: if } G = \emptyset \\
    &\quad \text{return } \tilde{C}_A \\
    &\text{6: let } S' = \bigcup_{i \in [k]: \tilde{c}_i^A \text{ is from a group in } G} L_i \text{ and } C' = \{\tilde{c}_i^A \in \tilde{C}_A : \tilde{c}_i^A \text{ is from a group not in } G\}; \\
    &\quad \text{recursively call Algorithm 4, where:} \\
    &\quad \quad S' \cup C' \cup C_0 \text{ plays the role of } S \\
    &\quad \quad \text{we assign elements in } C' \cup C_0 \text{ to an arbitrary group in } G \text{ and hence there are } |G| < m \text{ many groups } S_{j_1}, \ldots, S_{j_{|G|}} \\
    &\quad \quad \text{the requested numbers of centers are } k_{S_{j_1}}, \ldots, k_{S_{j_{|G|}}} \\
    &\quad \quad C' \cup C_0 \text{ plays the role of initially given centers } C_0 \\
    &\quad \text{let } \hat{C}_R \text{ denote its output} \\
    &\text{7: return } \hat{C}_R \cup C' \text{ as well as } (k_{S_j} - |C' \cap S_j|) \text{ many arbitrary elements from } S_j \text{ for every group } S_j \text{ not in } G
    \end{aligned}

**Usage Examples**
------------------
The Fair k-Center Clustering method can be applied in various scenarios where fair representation is crucial. For example:

- **Image Summarization**: Ensuring a balanced representation of different demographic groups in the summary of search results for a query like "CEO".

**Advantages and Limitations**
------------------------------

*Advantages:*

- Ensures fair representation of different demographic groups in the selected centers.
- The algorithm has a linear running time in the size of the data set and the number of centers, making it suitable for large datasets.

*Limitations:*

- The approximation guarantee of the algorithm may degrade as the number of demographic groups increases.
- The method requires knowledge of the group membership of each data point, which may not always be available.
- The recursive nature of the algorithm can lead to increased computational complexity in some cases.

**References**
---------------
1. Kleindessner, Matthäus, Pranjal Awasthi, and Jamie Morgenstern. "Fair k-center clustering for data summarization." International Conference on Machine Learning. PMLR, 2019.