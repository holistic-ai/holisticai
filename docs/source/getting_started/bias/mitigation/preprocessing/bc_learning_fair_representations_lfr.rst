Learning Fair Representations
-----------------------------

.. note::
    **Learning tasks:** Binary classification

Introduction
~~~~~~~~~~~~
The Learning Fair Representations (LFR) method aims to create a fair and accurate classification model by learning intermediate representations that mitigate bias with respect to sensitive attributes. The method ensures that the learned representations do not retain information about the membership in the protected group, thus promoting fairness in the classification decisions.

Description
~~~~~~~~~~~
- **Problem definition**
  The goal is to learn a representation :math:`Z` from the input data :math:`X` such that:
  
  1. The mapping from :math:`X` to :math:`Z` satisfies statistical parity.
  2. The mapping to :math:`Z` retains information in :math:`X` (except for membership in the protected set).
  3. The induced mapping from :math:`X` to :math:`Y` (by first mapping each :math:`x` probabilistically to :math:`Z`, and then mapping :math:`Z` to :math:`Y`) is close to the true classification function :math:`f`.

- **Main features**
  The LFR method is designed to achieve three main objectives:

  1. **Statistical Parity**: Ensures that the probability of mapping to any prototype is the same for both protected and unprotected groups.
  2. **Information Retention**: Encourages the representation to retain as much information as possible from the input data, except for the sensitive attribute.
  3. **Classification Accuracy**: Aims to achieve high accuracy in predicting the target variable.

- **Step-by-step description of the approach**

  1. **Prototype Assignment**: Each input example :math:`x` is stochastically assigned to a prototype :math:`v_k` using a softmax function:
     :math:`P(Z=k|x) = \frac{\exp(-d(x, v_k))}{\sum_{j=1}^{K} \exp(-d(x, v_j))}`
  2. **Objective Function**: The learning system minimizes the following objective:
     :math:`L = A_z \cdot L_z + A_x \cdot L_x + A_y \cdot L_y`
     where :math:`A_x`, :math:`A_y`, and :math:`A_z` are hyper-parameters governing the trade-off between the system desiderata.
  3. **Statistical Parity Term**: Ensures statistical parity by minimizing the difference in prototype assignments between protected and unprotected groups:
     :math:`L_z = \sum_{k=1}^{K} |M^+_k - M^-_k|`
     where :math:`M^+_k` and :math:`M^-_k` are the probabilities of mapping to prototype :math:`k` for the protected and unprotected groups, respectively.
  4. **Information Retention Term**: Measures the amount of information lost in the new representation using a squared-error measure:
     :math:`L_x = \sum_{n=1}^{N} (x_n - \hat{x}_n)^2`
     where :math:`\hat{x}_n` is the reconstruction of :math:`x_n` from :math:`Z`:
     :math:`\hat{x}_n = \sum_{k=1}^{K} M_{n,k} v_k`
  5. **Classification Accuracy Term**: Ensures accurate prediction of the target variable:
     :math:`L_y = \sum_{n=1}^{N} -y_n \log \hat{y}_n - (1 - y_n) \log (1 - \hat{y}_n)`
     where :math:`\hat{y}_n` is the prediction for :math:`y_n`, based on the prototype predictions:
     :math:`\hat{y}_n = \sum_{k=1}^{K} M_{n,k} w_k`
     and :math:`w_k` are the parameters governing the mapping from prototypes to class predictions.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Learning Fair Representation method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/preprocessing.html#3.-Learning-Fair-Representations>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.LearningFairRepresentation`.


References
~~~~~~~~~~~~~~
1. Zemel, Rich, et al. "Learning fair representations." International conference on machine learning. PMLR, 2013.