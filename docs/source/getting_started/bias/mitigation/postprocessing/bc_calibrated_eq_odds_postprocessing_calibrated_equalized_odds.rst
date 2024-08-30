Calibrated Equalized Odds Method
-----------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~~~~
The calibrated Equalized Odds method aims to address the inherent conflict between calibration and error-rate fairness in machine learning classifiers. Calibration ensures that predicted probabilities reflect true likelihoods, while Equalized Odds requires that error rates are balanced across different groups. This method explores a relaxation of Equalized Odds to achieve a feasible balance between these two fairness properties.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  The primary challenge addressed by the calibrated Equalized Odds method is the incompatibility between calibration and Equalized Odds. Calibration ensures that the predicted probabilities are accurate reflections of true probabilities, which is crucial for empirical risk analysis. Equalized Odds, on the other hand, ensures that no error type disproportionately affects any particular group. The method investigates the feasibility of relaxing Equalized Odds to maintain calibration while achieving a form of error-rate fairness.

- **Main features**

  The main features of the calibrated Equalized Odds method include:
  
  - A relaxation of the Equalized Odds conditions, requiring only that weighted sums of group error rates match.
  - A post-processing algorithm to achieve the optimal solution when the relaxed conditions are feasible.
  - An exploration of the trade-offs between calibration and error-rate fairness through empirical evaluations on various datasets.

- **Step-by-step description of the approach**

  1.  **Define Groups**: Identify and define distinct demographic groups :math:`G_t` relevant to the fairness consideration (e.g., gender, race).

  2. **Establish Equal Cost Constraint:** Determine a single cost function that balances false positive and false negative errors across all groups. This constraint ensures equitable treatment in terms of overall error rates.
  
  3.  **Train/Adjust Classifier**: Train or adjust the classifier to minimize the chosen equal-cost function while ensuring calibration within each group

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Calibrated Equalized Odds method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/postprocessing.html#1-.-Calibrated-Equalized-Odds>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.CalibratedEqualizedOdds`.

References
~~~~~~~~~~~~~~
1. Pleiss, Geoff, et al. “On fairness and calibration.” Advances in neural information processing systems 30 (2017).