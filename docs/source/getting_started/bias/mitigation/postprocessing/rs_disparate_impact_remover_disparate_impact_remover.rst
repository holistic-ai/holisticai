Disparate Impact Remover for Recommender Systems (RS)
-----------------------------------------------------

.. note::
    **Learning tasks:** Recommender systems.

Introduction
~~~~~~~~~~~~
The Disparate Impact Remover for Recommender Systems (RS) addresses the issue of unintended discrimination in algorithmic decision-making processes. Disparate impact occurs when a selection process yields significantly different outcomes for different groups, even if the process appears neutral. This method aims to quantify and remove disparate impact from datasets, ensuring that the resulting data can be used without causing unintended bias.

Description
~~~~~~~~~~~
- **Problem definition**

  The method tackles two main problems: the disparate impact certification problem and the disparate impact removal problem. The certification problem ensures that any classification algorithm applied to a dataset does not exhibit disparate impact. The removal problem involves transforming a dataset to eliminate disparate impact while preserving the ability to classify as much as possible.

- **Main features**

  The method is based on the 80% rule advocated by the US Equal Employment Opportunity Commission (EEOC). It involves measuring the conditional probability of a positive outcome for different protected attribute groups and ensuring that the ratio of these probabilities does not fall below a threshold of 0.8. The method also links disparate impact to the balanced error rate (BER) and uses a regression algorithm to minimize BER, thereby certifying the absence of disparate impact.

- **Step-by-step description of the approach**

  1. **Identify protected attributes and outcomes**: Given a dataset :math:`D = (X, Y, C)`, where :math:`X` represents protected attributes (e.g., race, sex, religion), :math:`Y` represents remaining attributes, and :math:`C` is the binary class to be predicted (e.g., "will hire"), identify the protected attribute and the positive outcome class.

  2. **Measure disparate impact**: Calculate the conditional probabilities :math:`Pr(C = \text{YES} | X = 0)` and :math:`Pr(C = \text{YES} | X = 1)`. If the ratio :math:`\frac{Pr(C = \text{YES} | X = 0)}{Pr(C = \text{YES} | X = 1)} \leq \tau = 0.8`, the dataset exhibits disparate impact.

  3. **Certify absence of disparate impact**: Use a regression algorithm that minimizes the balanced error rate (BER) to predict the protected attribute from the remaining attributes. If the BER is low, it indicates that the protected attribute can be predicted from the other attributes, suggesting the presence of disparate impact.

  4. **Transform the dataset**: Modify the dataset to make the protected attribute unpredictable from the remaining attributes. This involves altering the attributes :math:`Y` while preserving the class :math:`C` to the greatest extent possible.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Disparate impact remover for RS method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/recommender_systems/demos/postprocessing.html#Method:-Disparate-impact-remover>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.DisparateImpactRemoverRS`.

References
~~~~~~~~~~~~~~
1. Feldman, Michael, et al. "Certifying and removing disparate impact." Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. 2015.