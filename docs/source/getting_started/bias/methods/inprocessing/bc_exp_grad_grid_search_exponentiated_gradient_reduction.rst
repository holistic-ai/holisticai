Exponentiated Gradient Reduction Method
-------------------------------------------

.. note::
    **Learning tasks:** Binary classification, regression.

Introduction
~~~~~~~~~~~~~~~~
The Exponentiated Gradient (EG) reduction method is a technique used to achieve fairness in binary classification tasks. It is designed to optimize the tradeoff between accuracy and fairness by incorporating fairness constraints into the learning process. This method is particularly useful for ensuring that classifiers do not exhibit bias against protected attributes such as race or gender.

Description
~~~~~~~~~~~~~~~
The EG reduction method addresses the problem of fair classification by transforming it into a cost-sensitive classification problem. The main characteristics of this approach include:

- **Problem Definition:** The goal is to minimize classification error while satisfying fairness constraints, such as demographic parity or equalized odds.
- **Main Characteristics:** The method uses a Lagrangian formulation to incorporate fairness constraints into the objective function. It iteratively adjusts the costs associated with different training examples to achieve the desired fairness.
- **Step-by-Step Description:**

  1. **Formulate the Lagrangian:** Introduce Lagrange multipliers for each fairness constraint and form the Lagrangian function.
  2. **Saddle Point Problem:** Rewrite the problem as a saddle point problem, where the objective is to find a pair of solutions that minimize the Lagrangian with respect to the classifier and maximize it with respect to the Lagrange multipliers.
  3. **Iterative Algorithm:** Use an iterative algorithm to find the saddle point. The algorithm alternates between updating the classifier and the Lagrange multipliers.
  4. **Exponentiated Gradient Updates:** Use the exponentiated gradient algorithm to update the Lagrange multipliers, ensuring that they remain non-negative and sum to a bounded value.
  5. **Best Response Calculation:** At each iteration, calculate the best response of the classifier and the Lagrange multipliers.

References
~~~~~~~~~~~~~~
1. Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In Advances in Neural Information Processing Systems (pp. 656-666).