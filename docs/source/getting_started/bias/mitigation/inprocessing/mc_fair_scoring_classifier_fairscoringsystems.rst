FAIR Scoring System
------------------

.. note::
    **Learning tasks:** Multiclassification.

Introduction
~~~~~~~~~~~~
FAIR Scoring Classifier System is a Mixed Integer Linear Programming (MILP) framework designed to generate optimal scoring systems for multi-class classification tasks. The method ensures that the resulting models are interpretable, fair, and sparse. It incorporates fairness constraints to mitigate biases against protected groups and sparsity constraints to enhance model interpretability.

Description
~~~~~~~~~~~
FAIR Scoring Classifier System extends the Supersparse Linear Integer Model (SLIM) framework to handle multi-class classification problems. The method generates one scoring system for each class in a one-vs-all manner. The primary goal is to maximize classification accuracy while adhering to user-defined fairness and sparsity constraints.

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

Basic Usage
~~~~~~~~~~~~~~

The FAIR Scoring Classifier System method can be used as follows:

.. code-block:: python

  # Import the mitigator
  from holisticai.bias.mitigation import FairScoreClassifier

  # Create a FairScoreClassifier instance
  mitigator = FairScoreClassifier(**kargs)

  # Fit the mitigator on the training data
  mitigator.fit(train_data, y_data, group_a, group_b)

  # Predict using the mitigator on the test data
  test_data_transformed = mitigator.predict(test_data, group_a, group_b)

You can find an extended example of using the FAIR Scoring Classifier System method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/multi_classification/demos/inprocessing.html#1.-Fair-Scoring-Classifier>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.FairScoreClassifier`.

References
~~~~~~~~~~~~~~~~
1. Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for MultiClass Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial Intelligence, Oct 2022, Virtual, United States.