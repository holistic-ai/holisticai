Fair Regression with Wasserstein Barycenters
---------------------------------------------

.. note::
    **Learning tasks:** Regression.

Introduction
~~~~~~~~~~~~
The Wasserstein barycenter method addresses the problem of learning a real-valued regression function that satisfies the Demographic Parity constraint. This constraint requires the distribution of the predicted output to be independent of the sensitive attribute, ensuring fairness in predictions. The method establishes a connection between fair regression and optimal transport theory, leading to a closed-form expression for the optimal fair predictor. The optimal fair predictor's distribution is shown to be the Wasserstein barycenter of the distributions induced by the standard regression function on the sensitive groups. This result provides an intuitive interpretation of the optimal fair prediction and provides a simple post-processing algorithm to achieve fairness.

Description
~~~~~~~~~~~

- **Problem definition**

  The problem is to learn a real-valued regression function :math:`g` that minimizes the mean squared error while satisfying the Demographic Parity constraint. The constraint requires that the probability distribution of the predicted output is independent of the sensitive attribute :math:`S`. Formally, for every :math:`s, s' \in S`:

  .. math::
     \sup_{t \in \mathbb{R}} \left| P(g(X, S) \leq t | S = s) - P(g(X, S) \leq t | S = s') \right| = 0.

  This ensures that the Kolmogorov-Smirnov distance between the distributions :math:`\nu_{g|s}` and :math:`\nu_{g|s'}` vanishes for all :math:`s, s'`.

- **Main features**

  The main features of the method include:
  
  1. **Connection to Optimal Transport Theory**: The method leverages the theory of optimal transport, specifically the Wasserstein-2 distance, to derive the optimal fair predictor.
  2. **Closed-form Solution**: The optimal fair predictor is expressed in a closed form, making it computationally feasible to implement.
  3. **Post-processing Algorithm**: A simple post-processing procedure is provided, which can be applied to any off-the-shelf estimator to transform it into a fair predictor.
  4. **Fairness Guarantees**: The method provides distribution-free fairness guarantees and finite sample risk guarantees under certain conditions.

- **Step-by-step description of the approach**

  1. **Model Setup**: Consider the general regression model:

     .. math::
        Y = f^*(X, S) + \xi,

     where :math:`\xi` is a centered random variable, :math:`(X, S) \sim P_{X, S}` on :math:`\mathbb{R}^d \times S`, and :math:`f^*` is the regression function minimizing the squared risk.

  2. **Distribution of Predictions**: For any prediction rule :math:`f`, denote by :math:`\nu_{f|s}` the distribution of :math:`f(X, S) | S = s`. The Cumulative Distribution Function (CDF) of :math:`\nu_{f|s}` is given by:

     .. math::
        F_{\nu_{f|s}}(t) = P(f(X, S) \leq t | S = s).

  3. **Wasserstein-2 Distance**: The squared Wasserstein-2 distance between two univariate probability measures :math:`\mu` and :math:`\nu` is defined as:

     .. math::
        W_2^2(\mu, \nu) = \inf_{\gamma \in \Gamma_{\mu, \nu}} \int |x - y|^2 d\gamma(x, y),

     where :math:`\Gamma_{\mu, \nu}` is the set of distributions (couplings) on :math:`\mathbb{R} \times \mathbb{R}` such that for all :math:`\gamma \in \Gamma_{\mu, \nu}` and all measurable sets :math:`A, B \subset \mathbb{R}`, it holds that :math:`\gamma(A \times \mathbb{R}) = \mu(A)` and :math:`\gamma(\mathbb{R} \times B) = \nu(B)`.

  4. **Optimal Fair Predictor**: The optimal fair predictor :math:`g^*` is obtained by transforming the regression function :math:`f^*` as follows:

     .. math::
        g^*(x, s) = \left( \sum_{s' \in S} p_{s'} Q_{f^*|s'} \right) \circ F_{f^*|s}(f^*(x, s)),

     where :math:`p_s` is the frequency of group :math:`s`, and :math:`Q_{f^*|s}` and :math:`F_{f^*|s}` are the quantile and CDF functions of :math:`\nu_{f^*|s}`, respectively.

  5. **Summarizing the Post-processing Procedure**: The post-processing procedure involves the following steps:

     - Estimate the regression function :math:`f^*`.
     - Transform :math:`f^*` using the above formula to obtain an estimator of :math:`g^*`.
     - The transformation step requires only unlabeled data to estimate the cumulative distribution functions.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Wasserstein Barycenters method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/regression/examples/example_us_crime.html#Post-processing:-Wasserstein-Barycenters>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.WassersteinBarycenter`.

References
~~~~~~~~~~~~~~
1. Chzhen, Evgenii, et al. "Fair regression with wasserstein barycenters."Advances in Neural Information Processing Systems 33 (2020): 7321-7331.