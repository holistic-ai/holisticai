**Exponentiated Gradient Reduction Method**
===========================================

**Introduction**
----------------
The Exponentiated Gradient (EG) reduction method is a technique used to achieve fairness in binary classification tasks. It is designed to optimize the tradeoff between accuracy and fairness by incorporating fairness constraints into the learning process. This method is particularly useful for ensuring that classifiers do not exhibit bias against protected attributes such as race or gender.

**Description**
---------------
The EG reduction method addresses the problem of fair classification by transforming it into a cost-sensitive classification problem. The main characteristics of this approach include:

- **Problem Definition:** The goal is to minimize classification error while satisfying fairness constraints, such as demographic parity or equalized odds.
- **Main Characteristics:** The method uses a Lagrangian formulation to incorporate fairness constraints into the objective function. It iteratively adjusts the costs associated with different training examples to achieve the desired fairness.
- **Step-by-Step Description:**

  1. **Formulate the Lagrangian:** Introduce Lagrange multipliers for each fairness constraint and form the Lagrangian function.
  2. **Saddle Point Problem:** Rewrite the problem as a saddle point problem, where the objective is to find a pair of solutions that minimize the Lagrangian with respect to the classifier and maximize it with respect to the Lagrange multipliers.
  3. **Iterative Algorithm:** Use an iterative algorithm to find the saddle point. The algorithm alternates between updating the classifier and the Lagrange multipliers.
  4. **Exponentiated Gradient Updates:** Use the exponentiated gradient algorithm to update the Lagrange multipliers, ensuring that they remain non-negative and sum to a bounded value.
  5. **Best Response Calculation:** At each iteration, calculate the best response of the classifier and the Lagrange multipliers.

**Equations/Algorithms**
------------------------
The Lagrangian function is defined as:

.. math::
    :label: lagrangian

    L(Q, \lambda) = \hat{err}(Q) + \lambda^\top (M \hat{\mu}(Q) - \hat{c})

The saddle point problem is formulated as:

.. math::
    :label: saddle-point

    \min_{Q \in \Delta} \max_{\lambda \in \mathbb{R}^{|K|}_+, \|\lambda\|_1 \leq B} L(Q, \lambda)

**Usage Examples**
------------------
The EG reduction method was tested on several datasets to evaluate its performance in achieving fairness:

- **Adult Income Dataset:** Predicting whether someone makes more than $50k per year, with gender as the protected attribute.
- **COMPAS Recidivism Dataset:** Predicting recidivism from criminal history and demographics, with race as the protected attribute.
- **Law School Admissions Council's National Longitudinal Bar Passage Study:** Predicting bar exam passage, with race as the protected attribute.
- **Dutch Census Dataset:** Predicting whether someone has a prestigious occupation, with gender as the protected attribute.

**Advantages and Limitations**
------------------------------
*Advantages:*

- Can achieve any desired accuracy-fairness tradeoff.
- Works for any classifier representation.
- Encompasses many definitions of fairness.
- Does not require access to the protected attribute at test time.

*Limitations:*

- Requires access to the protected attribute at training time.
- May not achieve the lowest disparities on training data in some cases.

**References**
---------------
1. Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In Advances in Neural Information Processing Systems (pp. 656-666).