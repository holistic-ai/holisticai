CorrelationRemover
-------------------

.. note::
    **Learning tasks:** Binary classification, Multiclassification, Regression.

Introduction
~~~~~~~~~~~~~~~
The CorrelationRemover method is a pre-processing algorithm designed to mitigate unfairness in machine learning models by transforming input data. It specifically targets the removal of any correlation between input features and sensitive features, ensuring that the sensitive attributes do not influence the model's predictions.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  In many machine learning applications, input features may inadvertently carry information about sensitive attributes (e.g., race, gender). This correlation can lead to biased predictions and unfair outcomes. The CorrelationRemover method addresses this issue by applying a linear transformation to the input features, effectively removing any correlation with the sensitive features.

- **Main features**

  The main features of the CorrelationRemover method include:
  
  - **Linear Transformation**: Applies a linear transformation to the input features to eliminate correlation with sensitive features.
  - **Pre-processing Step**: Operates as a pre-processing step, transforming the data before it is passed to a standard training algorithm.

- **Step-by-step description of the approach**

  1. **Identify Sensitive Features**: The first step involves identifying the sensitive features in the dataset that need to be protected from influencing the model's predictions.
  
  2. **Compute Correlation**: Calculate the correlation between the input features and the sensitive features. This step quantifies the extent to which the input features are related to the sensitive attributes.
  
  3. **Apply Linear Transformation**: Use a linear transformation to adjust the input features, removing any detected correlation with the sensitive features while retaining much information as possible. This transformation ensures that the sensitive attributes do not influence the input features.

Basic Usage
~~~~~~~~~~~~~~

The CorrelationRemover method can be used as follows:

.. code-block:: python

  # Import the method
  from holisticai.bias.mitigation import CorrelationRemover

  # Create a CorrelationRemover instance
  mitigator = CorrelationRemover()

  # Transform the training data using the CorrelationRemover instance
  train_data_transformed = mitigator.fit_transform(train_data, group_a, group_b)

  # Transform the test data using the CorrelationRemover instance
  test_data_transformed = mitigator.transform(test_data, group_a, group_b)

You can find an extended example of using the CorrelationRemover method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/preprocessing.html#1-.-Correlation-Remover>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.CorrelationRemover`.
