Disparate Impact Remover
-------------------------

.. note::
    **Learning tasks:** Binary classification, Multiclassification, Regression, Clustering, Recommender systems.

Introduction
~~~~~~~~~~~~
The Disparate Impact Remover method addresses the issue of unintentional bias in algorithmic decision-making processes. Disparate impact occurs when a selection process yields significantly different outcomes for different groups, even if the process appears neutral. This method is particularly relevant in contexts where algorithms are used for critical decisions, such as hiring, lending, and sentencing, and aims to ensure fairness by modifying the data used by these algorithms.

Description
~~~~~~~~~~~

- **Problem definition**

  The problem of disparate impact is defined using the "80% rule" by the US Equal Employment Opportunity Commission (EEOC). Given a dataset :math:`D = (X, Y, C)`, where :math:`X` is a protected attribute (e.g., race, sex, religion), :math:`Y` represents the remaining attributes, and :math:`C` is the binary class to be predicted (e.g., "will hire"), the dataset is said to have disparate impact if:

  .. math::
      \frac{Pr(C = \text{YES} | X = 0)}{Pr(C = \text{YES} | X = 1)} \leq \tau = 0.8

  Here, :math:`Pr(C = c | X = x)` denotes the conditional probability that the class outcome is :math:`c` given the protected attribute :math:`x`.

- **Main features**

  The Disparate Impact Remover method focuses on two main problems:
  
  1. **Disparate Impact Certification**: Ensuring that any classification algorithm predicting some :math:`C'` from :math:`Y` does not exhibit disparate impact.
  2. **Disparate Impact Removal**: Modifying the dataset :math:`D` to produce a new dataset :math:`\bar{D} = (X, \bar{Y}, C)` that can be certified as not having disparate impact, while preserving the ability to classify as much as possible.

- **Step-by-step description of the approach**

  1. **Linking Disparate Impact to Classification Accuracy**:
     The method links disparate impact to the balanced error rate (BER), showing that any decision exhibiting disparate impact can be converted into one where the protected attribute can be predicted with low BER.

  2. **Certification Procedure**:
     A procedure is proposed to certify the impossibility of disparate impact on a dataset. This involves using a regression algorithm that minimizes BER, connecting BER to disparate impact in various settings (point and interval estimates, and distributions).

  3. **Data Transformation**:
     The input dataset is transformed to make the predictability of the protected attribute impossible. This transformation aims to preserve much of the signal in the unprotected attributes and maintain closeness to the original data distribution.

Basic Usage
~~~~~~~~~~~~~~

The Disparate Impact Remover method can be used as follows:

.. code-block:: python

  # Import the method
  from holisticai.bias.mitigation import DisparateImpactRemover

  # Create a DisparateImpactRemover instance
  mitigator = DisparateImpactRemover()

  # Transform the training data using the DisparateImpactRemover instance
  train_data_transformed = mitigator.fit_transform(train_data, group_a, group_b)

  # Transform the test data using the DisparateImpactRemover instance
  test_data_transformed = mitigator.transform(test_data, group_a, group_b)

You can find an extended example of using the DisparateImpactRemover method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/preprocessing.html#2.-Disparate-Impact-Remover>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.DisparateImpactRemover`.


References
~~~~~~~~~~~~~~
1. Feldman, Michael, et al. "Certifying and removing disparate impact." proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.