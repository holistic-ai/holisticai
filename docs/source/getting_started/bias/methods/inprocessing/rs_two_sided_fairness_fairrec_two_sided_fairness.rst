FairRec
-----------

.. note::
    **Learning tasks:** Recommendation systems.

Introduction
~~~~~~~~~~~~~~~~
FairRec is an algorithm designed to ensure two-sided fairness in personalized recommendation systems on two-sided platforms. It aims to balance the interests of both producers and customers by guaranteeing a minimum exposure for producers and ensuring that customers receive recommendations that are envy-free up to one item (EF1).

Description
~~~~~~~~~~~~~~~~
The FairRec algorithm addresses the problem of fair allocation in the context of recommendation systems. The goal is to allocate products to customers in a way that ensures fairness for both producers and customers. The main characteristics of FairRec include:

- **Producer Fairness**: Ensuring that each producer receives a minimum level of exposure, which is at least their maximin share (MMS) of exposure.
- **Customer Fairness**: Ensuring that the recommendations are envy-free up to one item (EF1) for all customers.

The algorithm operates in two phases:

1. **First Phase**: Ensures EF1 among all customers and tries to provide a minimum guarantee on the exposure of the producers.
2. **Second Phase**: Ensures that exactly k products are allocated to each customer while maintaining EF1 for customers.

References
~~~~~~~~~~~~~~~~
1. Patro, Gourab K., et al. "Fairrec: Two-sided fairness for personalized recommendations in two-sided platforms." Proceedings of The Web Conference 2020. 2020.