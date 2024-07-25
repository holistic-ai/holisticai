ρ-Fair Method
-----------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~~~~~
The ρ-Fair method is designed to address fairness in machine learning classification tasks. It aims to ensure that the classifier's performance is equitable across different groups defined by sensitive attributes. This method is significant as it provides a structured approach to mitigate biases and ensure fairness in predictive models.

Description
~~~~~~~~~~~~~~~~
The ρ-Fair method involves reducing the fairness problem to a series of Group-Fair problems, which are easier to solve. The main characteristics of the method include:

- **Problem Definition**: The goal is to find a classifier :math:`f` that minimizes prediction error while satisfying fairness constraints defined by a parameter :math:`\tau \in [0,1]`.
- **Main Characteristics**: The method uses a meta-algorithm that iteratively solves Group-Fair problems to approximate a solution for the ρ-Fair problem.
- **Step-by-Step Description**:

  1. **Estimate Distribution**: Compute an estimated distribution :math:`\hat{\mathcal{D}}` from the given samples.
  2. **Iterative Group-Fair Solutions**: For each iteration, define intervals :math:`a_i` and :math:`b_i` based on the fairness parameter :math:`\tau` and error parameter :math:`\epsilon`.
  3. **Compute Classifiers**: Solve the Group-Fair problem for each interval to obtain a set of classifiers.
  4. **Select Optimal Classifier**: Choose the classifier that minimizes the prediction error.

References
~~~~~~~~~~~~~~~~
1. Celis, L. Elisa, et al. "Classification with fairness constraints: A meta-algorithm with provable guarantees." Proceedings of the conference on fairness, accountability, and transparency. 2019.