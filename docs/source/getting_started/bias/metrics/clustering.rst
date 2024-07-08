Clustering
-----------

.. contents:: **Contents:**
    :depth: 2



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