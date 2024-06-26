
Bias Metrics
============

.. contents:: **Contents:**
    :depth: 2

Introduction
------------

Bias is defined as an unwanted prejudice in the decisions made by an AI system that systematically disadvantages a person or group. Various types of bias can exist and may be unintentionally introduced into algorithms at any stage of the development process, whether during data generation or model building.

    In order to measure whether a system treats different groups of people equally, we can follow two approaches: **equality of outcome** and **equality of opportunity**.

When we select **equality of outcome**, we ask that all subgroups have equal outcomes. For example, in a recruitment context, we may require that the percentage of applicants hired is consistent across groups (e.g. we want to hire 5% of all female applicants and 5% of all male applicants). Mathematically, this means that the likelihood of a positive outcome is equal for members of each group (regardless of the ground-truth labels):

.. math::
    P(\hat{Y} = 1 | G = a) = P(\hat{Y} = 1 | G = b) \quad \forall a, b

that is, the probability of a positive outcome is the same for all groups. 

When we select **equality of opportunity**, we ask that all subgroups are given the same opportunity of outcomes. For example, if we have a face recognition algorithm, we may want the classifier to perform equally well for all ethnicities and genders. Mathematically, the probability of a person in the positive class being correctly assigned a positive outcome and the probability of a person in a negative class being incorrectly assigned a positive outcome should both be the same for privileged and unprivileged group members. In this case, ground-truth labels are used to define the groups, and the following condition should hold:

.. math::
    P(\hat{Y} = 1 | Y = y, G = a) = P(\hat{Y} = 1 | Y = y, G = b) \quad \forall a, b ~~and~~ y \in \{0, 1\}

that is, the probability of a positive outcome is the same for all groups, given the ground-truth labels.

Binary Classification
---------------------------------------

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



Multi-Class Classification
---------------------------

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

Regression
-----------

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


Recommender Systems
---------------------

Equality of Outcome Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Mean Absolute Deviation:** Difference in average score for group a and group b.

    A large value of MAD indicates differential treatment of group a and group b. A positive value indicates that group a received higher scores on average, while a negative value indicates higher ratings for group b.

.. math::
    MAD = \frac{1}{n} \sum_{i=1}^{n} \hat{y}_{b} - \hat{y}_{a}

where :math:`\hat{y}_{g}` is the predicted score of group :math:`g`.

2. **Exposure Total Variation:**  This metric computes the total variation norm between the group a exposure distribution to the group b exposure distribution.

    A total variation divergence of 0 is desired, which occurs when the distributions are equal. The maximum value is 1 indicating the distributions are very far apart.

.. math::
    ETV = 0.5 \times \sum_{i=1}^{n} \left| item_{dist_{a}} - item_{dist_{b}} \right| \times n

where :math:`item_{dist_{g}}` is the distribution of items for group :math:`g`.


3. **Exposure KL Divergence:** This metric computes the KL divergence from the group a exposure distribution to the group_b exposure distribution.

    A KL divergence of 0 is desired, which occurs when the distributions are equal. Higher values of the KL divergence indicate difference in exposure distributions of group a and group b.

.. math::
    EKL = \sum_{i=1}^{n} item_{dist_{a}} \log \left( \frac{item_{dist_{a}}}{item_{dist_{b}}} \right)

where :math:`item_{dist_{g}}` is the distribution of items for group :math:`g`.


Equality of Opportunity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Average Precision Ratio:** This metric computes the ratio of average precision (over users) on group b and group a group.

    A value of 1 is desired. Lower values show bias against group b group. Higher values show bias against group a group.

.. math::
    APR = \frac{AP_{b}}{AP_{a}}

where :math:`AP_{g}` is the average precision of group :math:`g`.

2. **Average Recall Ratio:** This metric computes the ratio of average recall (over users) on group b and group a group.

    A value of 1 is desired. Lower values show bias against group b group. Higher values show bias against group a group.

.. math::
    Avg_{recall}=\frac{\text{AVG_recall_b}}{\text{AVG_recall_a}}

where :math:`\text{AVG_recall_g}` is the average recall of group :math:`g`.

3. **Average F1 Ratio:** This metric computes the ratio of average f1 (over users) on group b and group a group.

    A value of 1 is desired. Lower values show bias against group b group. Higher values show bias against group a group.

.. math::
    AFR = \frac{F1_{b}}{F1_{a}}

where :math:`F1_{g}` is the average f1 of group :math:`g`.

Item Metrics
~~~~~~~~~~~~

1. **Aggregate Diversity:** Given a matrix of scores, this metric computes the recommended items for each user, selecting either the highest-scored items or those above an input threshold. It then returns the aggregate diversity: the proportion of recommended items out of all possible items.

    A value of 1 is desired. We wish for a high proportion of items to be shown to avoid the 'rich get richer effect'.

.. math::
    \frac{|Items\; shown|}{|Items|}

where :math:`Items\; shown` is the number of items shown to users and :math:`Items` is the total number of items.

2. **GINI index:** Measures the inequality across the frequency distribution of the recommended items.

    An algorithm that recommends each item the same number of times (uniform distribution) will have a Gini index of 0 and the one with extreme inequality will have a Gini of 1.

.. math::
    GINI = \frac{\sum_{i=1}^{n} (2i - n - 1) \times item_{dist_{i}}}{n \times \sum_{i=1}^{n} item_{dist_{i}}}

where :math:`item_{dist_{i}}` is the distribution of items.

3. **Exposure Distribution Entropy:** This metric measures the entropy of the item exposure distribution.

    A low entropy (close to 0) indicates high certainty as to which item will be shown. Higher entropies therefore ensure a more homogeneous distribution. Scale is relative to number of items.

.. math::
    EDE = -\sum_{i=1}^{n} item_{dist_{i}} \log(item_{dist_{i}})

where :math:`item_{dist_{i}}` is the distribution of items.

4. **Average Recommendation Popularity:** This metric computes the average recommendation popularity of items over users. We define the recommendation popularity as the average amount of times an item is recommended.

    A low value is desidered and suggests that items have been recommended equally across the population.

.. math::
    ARP = \frac{1}{n} \sum_{i=1}^{n} item_{dist_{i}}

where :math:`item_{dist_{i}}` is the distribution of items.

Clustering
-----------

Equality of Outcome Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Social Fairness Ratio:** Given a centroid based clustering, this metric compute the average distance to the nearest centroid for both groups. The metric is the ratio of the resulting distance for group_a to group_b.

    A value of 1 is desired. Lower values indicate the group a is on average closer to the respective centroids. Higher values indicate that group_a is on average further from the respective centroids.

.. math::
    SFR = \frac{1}{n} \sum_{i=1}^{n} \frac{d_{a}}{d_{b}}

where :math:`d_{g}` is the average distance to the nearest centroid for group :math:`g`.

2. **Silhouette Difference:** We compute the difference of the mean silhouette score for both groups.

    The silhouette difference ranges from -1 to 1, with lower values indicating bias towards group a and larger values indicating bias against group b.

.. math::
    SD = \frac{1}{n} \sum_{i=1}^{n} \text{silhouette}_{b} - \text{silhouette}_{a}

where :math:`\text{silhouette}_{g}` is the mean silhouette score for group :math:`g`.


Equality of Opportunity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Cluster Balance:** Given a clustering and protected attribute. The cluster balance is the minimum over all groups and clusters of the ratio of the representation of members of that group in that cluster to the representation overall.

    A value of 1 is desired. That is when all clusters have the exact same representation as the data. Lower values imply the existence of clusters where either group a or group b is underrepresented.

.. math::
    CB = \min_{g, c} \left( \frac{N_{g,c}}{N_{c}} \right)

where :math:`N_{g,c}` is the number of members of group :math:`g` in cluster :math:`c` and :math:`N_{c}` is the total number of members in cluster :math:`c`.

2. **Minimum Cluster Ratio:** Given a clustering and protected attributes. The min cluster ratio is the minimum over all clusters of the ratio of number of group a members to the number of group b members.

    A value of 1 is desired. That is when all clusters are perfectly balanced. Low values imply the existence of clusters where group a has fewer members than group b.

.. math::
    MCR = \min_{c} \left( \frac{N_{a,c}}{N_{b,c}} \right)

where :math:`N_{g,c}` is the number of members of group :math:`g` in cluster :math:`c`.

3. **Cluster Distribution Total Variation:** This metric computes the distribution of group a and group b across clusters. It then outputs the total variation distance between these distributions.

    A value of 0 is desired. That indicates that both groups are distributed similarly amongst the clusters. The metric ranges between 0 and 1, with higher values indicating the groups are distributed in very different ways.

.. math::
    CDTV = 0.5 \times \sum_{i=1}^{n} \left| cluster_{dist_{a}} - cluster_{dist_{b}} \right| \times n

where :math:`cluster_{dist_{g}}` is the distribution of group :math:`g` across clusters.

4. **Cluster Distribution KL Div:** This metric computes the distribution of group a and group b membership across the clusters. It then returns the KL distance from the distribution of group a to the distribution of group b.

    A value of 0 is desired. That indicates that both groups are distributed similarly amongst the clusters. Higher values indicate the distributions of both groups amongst the clusters differ more.

.. math::
    CDKL = \sum_{i=1}^{n} cluster_{dist_{a}} \log \left( \frac{cluster_{dist_{a}}}{cluster_{dist_{b}}} \right)

where :math:`cluster_{dist_{g}}` is the distribution of group :math:`g` across clusters.

Summary Table
-------------

The following table summarizes the metrics that can be used to measure bias in different types of tasks.

.. csv-table:: Bias Metrics
    :header: "Class", "Task", "Metrics", "Ideal Value", "Fair Area", "Description"
    :file: bias_metrics.csv
