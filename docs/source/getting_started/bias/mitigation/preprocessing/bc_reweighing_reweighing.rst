Reweighing
-----------------

.. note::
    **Learning tasks:** Binary classification, Multiclassification.

Introduction
~~~~~~~~~~~~~~~
Reweighing is a method designed to address discrimination in datasets by adjusting the weights of data objects rather than altering their class labels. This approach is particularly useful in scenarios where maintaining the integrity of the original data is crucial. By assigning different weights to objects based on their sensitive attributes and class labels, Reweighing aims to eliminate bias while preserving the overall positive class probability. This method is less intrusive compared to other techniques, such as Massaging, which involves changing the labels of the objects.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  In many datasets, there exists a bias or discrimination against certain groups based on sensitive attributes such as gender, ethnicity, or age. This bias can lead to unfair treatment and inaccurate predictions by machine learning models. The goal of the Reweighing method is to mitigate this discrimination by adjusting the weights of the data objects, ensuring that the dataset becomes unbiased and that the learned classifier is discrimination-free.

- **Main features**

  The main features of the Reweighing method include:
  
  - **Non-intrusive adjustment**: Unlike methods that change the class labels of objects, Reweighing adjusts the weights assigned to each object, preserving the original data.
  - **Bias compensation**: By assigning weights based on the expected and observed probabilities of sensitive attribute and class label combinations, the method compensates for any existing bias.
  - **Discrimination-free classifier**: The adjusted weights ensure that the learned classifier is free from discrimination, providing fair and accurate predictions.

- **Step-by-step description of the approach**

  1. **Calculate Expected Probabilities**:
     For each combination of sensitive attribute :math:`S` and class label :math:`Class`, calculate the expected probability assuming no bias. This is done using the formula:
     
     .. math::

      P_{\text{exp}}(S=s \land Class=c) = \frac{| \{ X \in D | X(S) = s \} |}{|D|} \times \frac{| \{ X \in D | X(Class) = c \} |}{|D|}
     
     where :math:`D` is the dataset, :math:`s` is a specific value of the sensitive attribute, and :math:`c` is a specific class label.

  2. **Calculate Observed Probabilities**:
     Determine the observed probability for each combination of sensitive attribute and class label in the dataset:
     
     .. math::

      P_{\text{obs}}(S=s \land Class=c) = \frac{| \{ X \in D | X(S) = s \land X(Class) = c \} |}{|D|}
     

  3. **Compute Weights**:
     For each data object :math:`X`, compute its weight based on the ratio of the expected probability to the observed probability:
     
     .. math::

      W(X) = \frac{P_{\text{exp}}(S=X(S) \land Class=X(Class))}{P_{\text{obs}}(S=X(S) \land Class=X(Class))}
     
     This weight reflects the degree to which the object has been deprived or favored.

  4. **Assign Weights to Data Objects**:
     Assign the computed weights to the corresponding data objects in the dataset, creating a new weighted dataset :math:`D_W`.

  5. **Train Classifier**:
     Use the weighted dataset :math:`D_W` to train a classifier, taking into account the assigned weights. This ensures that the learned model is free from discrimination.

Basic Usage
~~~~~~~~~~~~~~

The Reweighing method can be used as follows:

.. code-block:: python

  # Import the method
  from holisticai.bias.mitigation import Reweighing

  # Create a Reweighing instance
  mitigator = Reweighing()

  # Transform the training data using the Reweighing instance
  train_data_transformed = mitigator.fit_transform(train_data, group_a, group_b)

  # Transform the test data using the Reweighing instance
  test_data_transformed = mitigator.transform(test_data, group_a, group_b)

You can find an extended example of using the Reweighing method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/preprocessing.html#4.-Reweighing>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.Reweighing`.

References
~~~~~~~~~~~~~~
1. Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques for classification without discrimination." Knowledge and information systems 33.1 (2012): 1-33.