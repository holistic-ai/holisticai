**FAIRScoringSystems**
======================

**Introduction**
----------------
FAIRScoringSystems is a Mixed Integer Linear Programming (MILP) framework designed to generate optimal scoring systems for multi-class classification tasks. The method ensures that the resulting models are interpretable, fair, and sparse. It incorporates fairness constraints to mitigate biases against protected groups and sparsity constraints to enhance model interpretability.

**Description**
---------------
FAIRScoringSystems extends the Supersparse Linear Integer Model (SLIM) framework to handle multi-class classification problems. The method generates one scoring system for each class in a one-vs-all manner. The primary goal is to maximize classification accuracy while adhering to user-defined fairness and sparsity constraints.

- **Problem Definition**: The method aims to create scoring systems that are both accurate and fair, ensuring that the classification does not disproportionately disadvantage any protected group.
- **Main Characteristics**:

  - **Fairness Constraints**: Ensures that the model's predictions are fair with respect to a chosen fairness metric and a specified tolerance level.
  - **Sparsity Constraints**: Limits the number of non-zero coefficients in the scoring system to enhance interpretability.
  - **Multi-Class Classification**: Extends binary scoring systems to multi-class problems using a one-vs-all approach.

- **Step-by-Step Description**:

  1. **Define Variables**:

     - Integer variables representing the coefficients of the scoring systems.
     - Binary loss variables indicating misclassification of training samples.
     - Binary variables indicating non-zero coefficients.

  2. **Set Up Objective Function**:

     - Maximize accuracy or balanced accuracy.

  3. **Incorporate Fairness Constraints**:

     - Define fairness metrics for multi-class classification.
     - Apply fairness constraints to sensitive labels.

  4. **Apply Sparsity Constraints**:

     - Limit the number of non-zero coefficients in each scoring system.

  5. **Solve MILP**:

     - Use an off-the-shelf MILP solver to find the optimal scoring systems.

**Equations/Algorithms**
------------------------

The objective function for maximizing accuracy is defined as:

.. math::
    :label: objective-accuracy

    \text{Maximize} \sum_{i=1}^{N} (1 - z_i)

where :math:`z_i` is a binary variable indicating misclassification of sample :math:`e_i`.

The fairness constraints for statistical parity are defined as:

.. math::
    :label: fairness-constraints

    \left| P(\hat{y}_i = k | e_i \in A) - P(\hat{y}_i = k | e_i \notin A) \right| \leq \epsilon_f \quad \forall k \in K_s

where :math:`\epsilon_f` is the unfairness tolerance, :math:`A` is the protected group, and :math:`K_s` is the set of sensitive labels.

**Usage Examples**
------------------
FAIRScoringSystems was tested on one synthetic and two real-world datasets:

- **Synthetic Dataset**: Features generated with a random distribution and labels computed with random noise. The dataset is biased towards certain labels for the sensitive feature.
- **Wine Dataset**: Chemical characteristics of wines associated with their quality. The sensitive feature is the color of the wine.
- **Customer Dataset**: Customer segmentation based on their profile. The sensitive feature is gender.

These datasets demonstrate the method's ability to produce interpretable models with good trade-offs between accuracy and fairness under different sparsity constraints.

**Advantages and Limitations**
------------------------------

*Advantages:*

- Produces interpretable models that are easy to understand and deploy.
- Ensures fairness by incorporating constraints based on popular fairness metrics.
- Allows for user-defined sparsity constraints to enhance model interpretability.
- Can handle multi-class classification problems effectively.

*Limitations:*

- Computationally intensive, especially for large datasets with many features.
- May require significant time to reach and prove optimality for complex datasets.
- The fairness constraints may lead to a trade-off with accuracy, potentially reducing overall performance.

**References**
---------------
1. Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for MultiClass Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial Intelligence, Oct 2022, Virtual, United States.