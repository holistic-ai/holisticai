Multi-Class Classification
---------------------------

.. contents:: **Contents:**
    :depth: 2



Equality of Outcome Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Multiclass Statistical Parity:**  This metric computes statistical parity for a classification task with multiple classes and a protected attribute with multiple groups. For each group compute the vector of success rates for entering each class. Compute all distances (mean absolute deviation) between such vectors. Then aggregate them using the mean, or max strategy.

    The accepted values and bounds for this metric are the same as the 1d case. A value of 0 is desired. Values below 0.1 are considered fair.

.. math::
    SP_{max} = \max_{g} \left| SR_{g} - SR_{a} \right|

where :math:`SR_{g}` is the ratio of the number of positive outcomes to the total number of outcomes in that group.

If the mean strategy is selected, the metric is defined as:

.. math::
    SP_{mean} = \frac{1}{n} \sum_{i=1}^{n} \left| SR_{g} - SR_{a} \right|


Equality of Opportunity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Multiclass Equality of Opportunity:** This metric is a multiclass generalisation of Equality of Opportunity. For each group, compute the matrix of error rates (normalised confusion matrix). Compute all distances (mean absolute deviation) between such matrices. Then aggregate them using the mean, or max strategy.

    The accepted values and bounds for this metric are the same as the 1d case. A value of 0 is desired. Values below 0.1 are considered fair.

.. math::
    EOD_{max} = \max_{g} \left| TPR_{g} - TPR_{a} \right|

where :math:`TPR_{g}` is the true positive rate of group :math:`g`.

If the mean strategy is selected, the metric is defined as:

.. math::
    EOD_{mean} = \frac{1}{n} \sum_{i=1}^{n} \left| TPR_{g} - TPR_{a} \right|


2. **Multiclass Average Odds:** This metric is a multiclass generalisation of Average Odds. For each group, compute the matrix of error rates (normalised confusion matrix). Average these matrices over rows, and compute all pariwise distance (mean absolute deviation) between the resulting vectors. Aggregate results using either mean or max strategy.

    The accepted values and bounds for this metric are the same as the 1d case. A value of 0 is desired. Values below 0.1
    are considered fair.

.. math::
    AOD_{max} = \max_{g} \left| \frac{1}{2}[(TPR_{g} - TPR_{a}) + (FPR_{g} - FPR_{a})] \right|

where :math:`TPR_{g}` is the true positive rate of group :math:`g` and :math:`FPR_{g}` is the false positive rate of group :math:`g`.

If the mean strategy is selected, the metric is defined as:

.. math::
    AOD_{mean} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{1}{2}[(TPR_{g} - TPR_{a}) + (FPR_{g} - FPR_{a})] \right|

3. **Multiclass True Positive Difference:** This metric is a multiclass generalisation of TPR Difference. For each group, compute the matrix of error rates (normalised confusion matrix). Compute all distances (mean absolute deviation) between the diagonals of such matrices. Then aggregate them using the mean, or max strategy.

    The accepted values and bounds for this metric are the same as the 1d case. A value of 0 is desired. Values below 0.1 are considered fair.

.. math::
    TPD_{max} = \max_{g} \left| TPR_{g} - TPR_{a} \right|

where :math:`TPR_{g}` is the true positive rate of group :math:`g`.

If the mean strategy is selected, the metric is defined as:

.. math::
    TPD_{mean} = \frac{1}{n} \sum_{i=1}^{n} \left| TPR_{g} - TPR_{a} \right|