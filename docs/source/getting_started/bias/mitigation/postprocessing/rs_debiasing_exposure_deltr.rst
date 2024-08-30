DELTR
-----------------

Introduction
~~~~~~~~~~~~~~~
DELTR is an in-processing approach that extends ListNet with a list-wise fairness objective. This method aims to reduce the extent to which protected elements receive less exposure in ranking systems. The significance of DELTR lies in its ability to achieve list-wise fairness without necessarily compromising accuracy. In fact, aiming for list-wise fairness can increase relevance in cases corresponding to non-discrimination. DELTR is designed to handle different types of biases without prior knowledge about their nature, making it a versatile tool for achieving fairer ranking outcomes.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  The primary problem addressed by DELTR is the disparate exposure of protected groups in ranking systems. Traditional Learning to Rank (LTR) models can reproduce and even exaggerate discrepancies in visibility between protected and non-protected groups. DELTR aims to mitigate this issue by incorporating a fairness objective into the ranking process, ensuring that protected elements receive fair exposure without sacrificing the overall relevance of the rankings.

- **Main features**

  DELTR extends the ListNet algorithm by adding a list-wise fairness objective. This objective is designed to balance the exposure of protected and non-protected groups. The method is flexible, allowing for the adjustment of the parameter :math:`\gamma`, which controls the trade-off between relevance and fairness. DELTR can handle different types of biases, whether they require the inclusion or exclusion of protected features during training. This flexibility makes DELTR a robust solution for achieving fairer rankings across various datasets and scenarios.

- **Step-by-step description of the approach**

  1. **Initialization**: DELTR starts by initializing the ListNet algorithm with an additional list-wise fairness objective. This objective is designed to ensure that protected elements receive fair exposure in the ranking results.

  2. **Parameter Setting**: The parameter :math:`\gamma` is introduced to control the trade-off between relevance and fairness. Two values of :math:`\gamma` are typically used: :math:`\gamma_{\text{small}}`, which is an order of magnitude smaller than the standard loss :math:`L`, and :math:`\gamma_{\text{large}}`, which is comparable to the value of :math:`L`.

  3. **Training**: During the training phase, DELTR optimizes both the relevance and fairness objectives simultaneously. This involves adjusting the ranking model to balance the exposure of protected and non-protected groups while maintaining high relevance.

  4. **Evaluation**: The performance of DELTR is evaluated against several baselines, including:
     - A "colorblind" LTR approach that excludes protected attributes during training.
     - A standard LTR method that includes protected attributes during training.
     - A post-processing approach that re-ranks the output after applying LTR.
     - A pre-processing approach that modifies the training data before applying LTR.
