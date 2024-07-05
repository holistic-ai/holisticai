Regression
-----------

.. contents:: **Contents:**
    :depth: 2


Equality of Outcome Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **q-Disparate Impact:** This metric computes the ratio of success rates between group a and group b, where sucess means predicted score exceeds a given quantile (default = 0.8).If q is a vector, this metric returns a vector with the respective result for each given quantile in q.
    
    A value of 1 is desired. Values below 1 are unfair towards group_a. Values above 1 are unfair towards group_b. The range (0.8,1.2)is considered acceptable.

.. math::
    DI_{q} = \frac{SR_{b}}{SR_{a}}

where :math:`SR_{g}` is the ratio of the number of positive outcomes to the total number of outcomes in that group.

2. **No Disparate Impact Level:** If we calculate the adverse impact for each possible quantile, we can find the minimum maximum quantile for which the algorithm is considered unbiased (i.e. the disparate impact falls between 0.8 and 1.2).

.. math::
    NoDI = \min_{q} \{q \in [0, 1] : DI_{q} \in [0.8, 1.2]\}


3. **Average Score Difference:** this metric is the difference between the average score of the unprivileged and privileged group. 

    The ideal value is 0, a value < 0 disadvantages the unprivileged group and > 0 is favorable.

.. math::
    ASD = \frac{1}{n} \sum_{i=1}^{n} \hat{y}_{b} - \hat{y}_{a}

where :math:`\hat{y}_{g}` is the predicted score of group :math:`g`.


4. **Average Score Ratio:** this metric computes the ratio in average scores between group a and group b. If q is a vector, this metric returns a vector with the respective result for each given quantile in q.

    A value of 1 is desired. Values below 1 indicate the group a has lower average score, so bias against group_a. Values above 1 indicate group_b has lower average score, so bias against group_b. The [0.8, 1.25] range is considered fair.

.. math::
    ASR = \frac{\hat{y}_{b}}{\hat{y}_{a}}

where :math:`\hat{y}_{g}` is the predicted score of group :math:`g`.


5. **Z Score Difference:** the Z score spread is the average score spread divided by the pooled standard deviation. It allows us to compare the difference in average scores with the standard deviation. 
    
    The ideal value is 0, a value less than 0 disadvantages the unprivileged group and larger than 0 is favorable.

.. math::
    ZSD = \frac{1}{n} \sum_{i=1}^{n} \frac{\hat{y}_{b} - \hat{y}_{a}}{poolStd}

where :math:`poolStd` is the pooled standard deviation of the predicted scores, defined as


6. **Max Statistical Parity:** This metric computes the maximum over all thresholds of the absolute statistical parity between group a and group b.

    A value of 0 is desired. Values below 0.1 in absolute value are considered acceptable.

.. math::
    SP_{max} = \max_{t} \left| SR_{b} - SR_{a} \right|

where :math:`SR_{g}` is the ratio of the number of positive outcomes to the total number of outcomes in that group.


7. **Statistical Parity AUC:** This metric computes the area under the statistical parity versus threshold curve. 
    
    A value of 0 is desired. Values below 0.075 are considered acceptable.

.. math::
    SPAUC = \int_{0}^{1} \left| SR_{b} - SR_{a} \right| dt

where :math:`SR_{g}` is the ratio of the number of positive outcomes to the total number of outcomes in that group.


Equality of Opportunity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **RMSE Ratio:** This metric computes the ratio of the RMSE for group a and group b. If q is a vector, this metric returns a vector with the respective result for each given quantile in q.

    A value of 1 is desired. Lower values show bias against group a. Higher values show bias against group b.

.. math::
    RMSE_{ratio} = \frac{RMSE_{b}}{RMSE_{a}}

where :math:`RMSE_{g}` is the root mean squared error of group :math:`g`.

2. **MAE Ratio:** This metric computes the ratio of the MAE for group a and group b. If q is a vector, this metric returns a vector with the respective result for each given quantile in q.

    A value of 1 is desired. Lower values show bias against group a. Higher values show bias against group b.

.. math::
    MAE_{ratio} = \frac{MAE_{b}}{MAE_{a}}

where :math:`MAE_{g}` is the mean absolute error of group :math:`g`.

3. **Correlation Difference:** This metric computes the difference in correlation between predictions and targets for group a and group b. If q is a vector, this metric returns a vector with the respective result for each given quantile in q.

    A value of 0 is desired. This metric ranges between -2 and 2, with -1 indicating strong bias against group a, and +1 indicating strong bias against group b.

.. math::
    CD = \rho_{b} - \rho_{a}

where :math:`\rho_{g}` is the correlation between predictions and targets for group :math:`g`.