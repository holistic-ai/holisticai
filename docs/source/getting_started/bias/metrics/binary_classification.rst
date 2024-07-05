
Binary Classification
----------------------

.. contents:: **Contents:**
    :depth: 2

Equality of Outcome Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The idea of equality of outcome metrics in classification tasks, is to compare the rate of success in the privileged group with the rate of success in the unprivileged group. We define the success rate :math:`SR_{g}` of a group :math:`g` as the ratio of the number of positive outcomes to the total number of outcomes in that group:

.. math::
    SR_{g} = \frac{TP_{g}}{TP_{g} + FP_{g}}

where :math:`TP_{g}` is the number of true positives in group :math:`g` and :math:`FP_{g}` is the number of false positives in group :math:`g`.

.. note::
    The idea is that an unbiased system would present roughly similar success rates across groups. 

We will refer to the success rate for the unprivileged group as :math:`SR_{b}` and the success rate for the privileged group as :math:`SR_{a}`. We can then define the following metrics.

1. **Disparate Impact (DI):** measures the ratio of success rates. 

    The ideal value is 1. The acceptable range is [0.8, 1.2]. Values below 0.8 are unfair towards group_a. Values above 1.2 are unfair towards group_b.

.. math::
    DI = \frac{SR_{b}}{SR_{a}}

2. **Statistical Parity (SP):** measures the difference between success rates. A negative value means that the unprivileged group_b is unfavoured. 

    Ideal value is 0. Negative values are unfair towards group b.

.. math::
    SP = SR_{b} - SR_{a}

3. **Cohen's D (CD):** measures the effect size of the difference between success rates. 

    Ideal value is 0. Positive values are unfair towards group b. 
    Reference values: 0.2 is considered a small effect size, 0.5 is considered medium, 0.8 is considered large.

.. math::
    CD = \frac{SR_{b} - SR_{a}}{poolStd}

where :math:`poolStd` is the pooled standard deviation of the success rates, defined as

.. math::
    poolStd = \frac{(n_{a} - 1)\sigma^{2}_{a} + (n_{b} - 1)\sigma^{2}_{b}}{n_{a} + n_{b} - 2}

where :math:`n_{a}` and :math:`n_{b}` are the number of samples in groups :math:`a` and :math:`b`, respectively, and :math:`\sigma^{2}_{a}` and :math:`\sigma^{2}_{b}` are the variances of the success rates in groups :math:`a` and :math:`b`, respectively.

4. **2-SD Rule:** measures the difference between success rates in terms of standard deviations. 

    The ideal value is 0. Positive values are unfair towards group b.

.. math::
    2-SD = \frac{SR_{b} - SR_{a}}{\sqrt{\frac{SR_{a}(1 - SR_{a})}{n_{a}} + \frac{SR_{b}(1 - SR_{b})}{n_{b}}}}

5. **Four-Fifths Rule:** measures the ratio of success rates. 

    The ideal value is 1. Values below 0.8 and above 1.2 are considered unfair towards group_b.

.. math::
    Four-Fifths = \frac{SR_{b}}{SR_{a}} \geq 0.8


Equality of Opportunity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The idea of equality of opportunity metrics, is to compare true positives and/or false positives rates across groups. We define the true positive rate :math:`TPR_{g}` of a group :math:`g` as the ratio of the number of true positives to the total number of actual positives in that group:

.. math::
    TPR_{g} = \frac{TP_{g}}{TP_{g} + FN_{g}}

where :math:`FN_{g}` is the number of false negatives in group :math:`g`.

1. **Equality of Opportunity Difference:** measures the difference between true positive rates. Ideal value: 0 and Fair area: [-0.1, 0.1]

.. math::
    EOD = TPR_{b} - TPR_{a}

where :math:`TPR_{g}` is the true positive rate of group :math:`g`.

2. **False Positive Rate Difference:** measures the difference between false positive rates. :

    The ideal value is 0. Positive values are unfair towards group b.

.. math::
    FPRD = FPR_{b} - FPR_{a}

where :math:`FPR_{g}` is the false positive rate of group :math:`g`.


3. **Average Odds Difference** measures the average of the difference between true positive rates and false positive rates. Ideal value: 0 and Fair area: [-0.1, 0.1]

.. math::
    AOD = \frac{1}{2}[(TPR_{b} - TPR_{a}) + (FPR_{b} - FPR_{a})]


4. **Accuracy Difference:** measures the difference between accuracy rates. 

    The ideal value is 0. Positive values are unfair towards group b.

.. math::
    AD = ACC_{b} - ACC_{a}

where :math:`ACC_{g}` is the accuracy of group :math:`g`.