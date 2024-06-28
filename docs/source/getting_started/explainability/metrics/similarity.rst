Order Cohesion Metrics
======================

Position Parity and Rank Alignment are metrics designed to evaluate the similarity between feature importance rankings under different conditions or groups. These metrics help in understanding how consistent the feature rankings are across different subgroups, providing insights into model robustness and reliability.

.. contents:: Table of Contents
   :local:
   :depth: 1

Position Parity
---------------

Position Parity measures the consistency of the order of feature importance between the overall feature importance and the conditional feature importance for different groups.

Methodology
~~~~~~~~~~~

1. **Compute Match Order:**
   - For each group, compare the order of feature importance with the overall feature importance and calculate the cumulative match order.

2. **Calculate Position Parity:**
   - Compute the mean of the cumulative match orders for each group and average them to obtain the Position Parity Score.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( C \) denote the conditional feature importance and \( R \) denote the overall feature importance. For each group \( g \), the match order \( M \) is given by:

.. math::

   M_{g}(i) = \begin{cases} 
      1 & \text{if } C_{g,i} = R_i \\
      0 & \text{otherwise}
   \end{cases}

The cumulative match order \( M_{g,\text{cum}} \) is:

.. math::

   M_{g,\text{cum}}(i) = \frac{\sum_{j=1}^{i} M_{g}(j)}{i}

The Position Parity Score \( PP \) is the mean of the cumulative match orders:

.. math::

   PP = \frac{1}{|G|} \sum_{g \in G} \frac{1}{n} \sum_{i=1}^{n} M_{g,\text{cum}}(i)

where \( G \) is the set of all groups and \( n \) is the number of features.

Interpretation
~~~~~~~~~~~~~~

- **High Score:** Indicates high consistency in the order of feature importance between the overall and conditional feature importance, suggesting that the model behaves similarly across different groups.
- **Low Score:** Indicates low consistency in the order of feature importance, suggesting variability in model behavior across different groups.

Rank Alignment
--------------

Rank Alignment measures the overlap in the top-k feature importance rankings between the overall feature importance and the conditional feature importance for different groups.

Methodology
~~~~~~~~~~~

1. **Compute Intersections:**
   - For each group and for each top-k subset, calculate the intersection between the top-k features of the conditional feature importance and the overall feature importance.

2. **Calculate Rank Alignment:**
   - Compute the mean of the intersection ratios for each group and average them to obtain the Rank Alignment Score.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( C \) denote the conditional feature importance and \( R \) denote the overall feature importance. For each group \( g \), the intersection ratio \( I \) for top-k features is given by:

.. math::

   I_{g,k} = \frac{| \text{top-k}(C_{g}) \cap \text{top-k}(R) |}{k}

The Rank Alignment Score \( RA \) is the mean of the intersection ratios:

.. math::

   RA = \frac{1}{|G|} \sum_{g \in G} \frac{1}{n} \sum_{k=1}^{n} I_{g,k}

where \( G \) is the set of all groups and \( n \) is the number of features.

Interpretation
~~~~~~~~~~~~~~

- **High Score:** Indicates a high overlap in the top-k feature importance rankings between the overall and conditional feature importance, suggesting that the model's feature importance is consistent across different groups.
- **Low Score:** Indicates a low overlap in the top-k feature importance rankings, suggesting variability in the model's feature importance across different groups.
