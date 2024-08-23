Exponentiated Gradient Reduction Method
---------------------------------------

.. note::
    **Learning tasks:** Binary classification, regression.

Introduction
~~~~~~~~~~~~
The Exponentiated Gradient (EG) reduction method is a technique used to achieve fairness in binary classification tasks. It is designed to optimize the tradeoff between accuracy and fairness by incorporating fairness constraints into the learning process. This method is particularly useful for ensuring that classifiers do not exhibit bias against protected attributes such as race or gender.

Description
~~~~~~~~~~~
The EG reduction method addresses the problem of fair classification by transforming it into a cost-sensitive classification problem. The main characteristics of this approach include:

- **Problem Definition:** The goal is to minimize classification error while satisfying fairness constraints, such as demographic parity or equalized odds.
- **Main Characteristics:** The method uses a Lagrangian formulation to incorporate fairness constraints into the objective function. It iteratively adjusts the costs associated with different training examples to achieve the desired fairness.
- **Step-by-Step Description:**

  1. **Formulate the Lagrangian:** Introduce Lagrange multipliers for each fairness constraint and form the Lagrangian function.
  2. **Saddle Point Problem:** Rewrite the problem as a saddle point problem, where the objective is to find a pair of solutions that minimize the Lagrangian with respect to the classifier and maximize it with respect to the Lagrange multipliers.
  3. **Iterative Algorithm:** Use an iterative algorithm to find the saddle point. The algorithm alternates between updating the classifier and the Lagrange multipliers.
  4. **Exponentiated Gradient Updates:** Use the exponentiated gradient algorithm to update the Lagrange multipliers, ensuring that they remain non-negative and sum to a bounded value.
  5. **Best Response Calculation:** At each iteration, calculate the best response of the classifier and the Lagrange multipliers.

Basic Usage
~~~~~~~~~~~~~~

The Exponentiated Gradient Reduction method can be used as follows:

.. code-block:: python

  # Import the mitigator
  from holisticai.bias.mitigation import ExponentiatedGradientReduction

  # Create a ExponentiatedGradientReduction instance
  mitigator = ExponentiatedGradientReduction(**kargs)

  # Fit the mitigator on the training data
  mitigator.fit(train_data, y_data, group_a, group_b)

  # Predict using the mitigator on the test data
  test_data_transformed = mitigator.predict(test_data)

You can find an extended example of using the Exponentiated Gradient Reduction method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/inprocessing.html#2.-Exponentiated-Gradient>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.ExponentiatedGradientReduction`.

References
~~~~~~~~~~
1. Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In Advances in Neural Information Processing Systems (pp. 656-666).