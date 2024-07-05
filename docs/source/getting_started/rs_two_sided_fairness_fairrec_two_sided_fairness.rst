**FairRec**
=================

**Introduction**
----------------
FairRec is an algorithm designed to ensure two-sided fairness in personalized recommendation systems on two-sided platforms. It aims to balance the interests of both producers and customers by guaranteeing a minimum exposure for producers and ensuring that customers receive recommendations that are envy-free up to one item (EF1).

**Description**
---------------
The FairRec algorithm addresses the problem of fair allocation in the context of recommendation systems. The goal is to allocate products to customers in a way that ensures fairness for both producers and customers. The main characteristics of FairRec include:

- **Producer Fairness**: Ensuring that each producer receives a minimum level of exposure, which is at least their maximin share (MMS) of exposure.
- **Customer Fairness**: Ensuring that the recommendations are envy-free up to one item (EF1) for all customers.

The algorithm operates in two phases:

1. **First Phase**: Ensures EF1 among all customers and tries to provide a minimum guarantee on the exposure of the producers.
2. **Second Phase**: Ensures that exactly k products are allocated to each customer while maintaining EF1 for customers.

**Equations/Algorithms**
------------------------
The key equation used in FairRec is as follows:

- **Maximin Share (MMS)**: The maximin share threshold for an agent :math:`u` is defined as:

  .. math::
      :label: eq-mms

      \text{MMS}_u = \max_{A} \min_{w \in U} V_u(A_w)


**Usage Examples**
------------------
FairRec has been tested on multiple real-world datasets, including:

- **Google Local (GL-CUSTOM)**: A dataset containing customer reviews and ratings for local businesses.
- **Google Local (GL-FACT)**: A dataset with factual information about local businesses.
- **Last.fm (LF)**: A dataset containing user interactions with music tracks.

In these datasets, FairRec demonstrated its effectiveness in ensuring two-sided fairness while maintaining a high level of recommendation quality.

**Advantages and Limitations**
------------------------------
*Advantages:*

- Ensures a minimum exposure guarantee for producers, promoting fairness and economic opportunities.
- Maintains customer fairness by ensuring recommendations are envy-free up to one item (EF1).
- Scalable and adaptable to different data-driven models for estimating product-customer relevance scores.

*Limitations:*

- May incur a marginal loss in overall recommendation quality due to the fairness constraints.
- The exact value of the minimum exposure guarantee (E) needs to be decided by the platform, which may vary based on specific requirements and contexts.

**References**
---------------
1. Patro, Gourab K., et al. "Fairrec: Two-sided fairness for personalized recommendations in two-sided platforms." Proceedings of The Web Conference 2020. 2020.