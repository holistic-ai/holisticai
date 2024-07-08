Recommender Systems
---------------------

.. contents:: **Contents:**
    :depth: 2



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