Equality of Opportunity
-----------------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~
The Equality of Opportunity method is a fairness criterion proposed to address discrimination against specified sensitive attributes in supervised learning. This method aims to predict a target based on available features while ensuring that the prediction process does not unfairly disadvantage any protected group. The method is significant as it shifts the cost of poor classification from disadvantaged groups to the decision maker, thereby incentivizing the improvement of classification accuracy.

Description
~~~~~~~~~~~
- **Problem definition**

  The problem addressed by the Equality of Opportunity method is the potential discrimination in supervised learning algorithms against protected groups. The goal is to adjust any learned predictor to remove discrimination according to the defined fairness criterion. This involves ensuring that the probability of a positive outcome is equal for individuals who would actually achieve the positive outcome, regardless of their membership in a protected group.

- **Main features**

  The main features of the Equality of Opportunity method include:
  
  - **Fairness Criterion**: The method ensures that the true positive rate is equal across different groups defined by the sensitive attribute.
  - **Incentive Structure**: It creates an incentive for the decision maker to improve the classification accuracy by shifting the cost of poor classification from the disadvantaged groups to the decision maker.
  - **Post-Processing Step**: The method can be implemented as a post-processing step, which adjusts the predictions of an existing classifier without requiring changes to the training process.
  - **Oblivious Nature**: The method relies only on the joint statistics of the predictor, the target, and the protected attribute, without requiring interpretation of individual features.

- **Step-by-step description of the approach**

  1. **Initial Predictor**: Start with an existing learned binary predictor :math:`\hat{Y}` or a score :math:`R` that has been trained on the available data.
  
  2. **Derived Predictor**: Construct a derived predictor :math:`\tilde{Y}` from the random variable :math:`R` and the protected attribute :math:`A`. The derived predictor :math:`\tilde{Y}` is a possibly randomized function of :math:`(R, A)` alone and is independent of the features :math:`X` conditional on :math:`(R, A)`.
  
  3. **Equalized Odds**: Ensure that the derived predictor satisfies the equalized odds criterion, which requires that the true positive rate and the false positive rate are equal across different groups defined by the protected attribute.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Equality of Opportunity method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/postprocessing.html#2.-Equalized-Odds>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.EqualizedOdds`.

References
~~~~~~~~~~~~~~
1. Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016).