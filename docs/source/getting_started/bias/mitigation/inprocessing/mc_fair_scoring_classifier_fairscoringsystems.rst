FAIRScoringSystems
------------------

.. note::
    **Learning tasks:** Multiclassification.

Introduction
~~~~~~~~~~~~~~~~
FAIRScoringSystems is a Mixed Integer Linear Programming (MILP) framework designed to generate optimal scoring systems for multi-class classification tasks. The method ensures that the resulting models are interpretable, fair, and sparse. It incorporates fairness constraints to mitigate biases against protected groups and sparsity constraints to enhance model interpretability.

Description
~~~~~~~~~~~~~~~~
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

References
~~~~~~~~~~~~~~~~
1. Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for MultiClass Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial Intelligence, Oct 2022, Virtual, United States.