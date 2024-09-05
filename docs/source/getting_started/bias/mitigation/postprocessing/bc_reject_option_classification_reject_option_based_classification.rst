Reject Option based Classification (ROC)
----------------------------------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~
The Reject Option based Classification (ROC) method is designed to address the issue of discrimination in classification tasks. Discrimination in this context refers to biased decision-making based on sensitive attributes such as gender or race. The ROC method leverages the concept of posterior probabilities from probabilistic classifiers to identify and mitigate discriminatory decisions. 

Description
~~~~~~~~~~~
The ROC method is an approach that deviates from traditional classification rules by incorporating a reject option for instances with uncertain classifications. By introducing a critical region where decisions are uncertain, the method aims to neutralize the effect of biases, ensuring fairer outcomes without significantly compromising accuracy. The ROC method introduces the concept of a critical region where the classification decision is uncertain. Instances falling within this region are relabeled to reduce discrimination. 

- **Problem definition**

  The problem addressed by ROC is the presence of discrimination in classification tasks. Given a dataset :math:`\mathcal{D} = \{(X_i, C_i)\}_{i=1}^N` where :math:`C_i` are the class labels and :math:`X_i` are the instances described by a set of attributes, some of which are sensitive (e.g., gender, race), the goal is to learn a classifier :math:`\mathcal{F}: \mathcal{X} \rightarrow \{C^+, C^-\}` that does not make discriminatory decisions based on these sensitive attributes.

- **Main features**

  The main features of the ROC method include:
  
  - Utilization of posterior probabilities from probabilistic classifiers.
  - Introduction of a critical region defined by a threshold :math:`\theta`.
  - Relabeling of instances in the critical region based on group membership to mitigate discrimination.
  - Flexibility to work with single or multiple probabilistic classifiers.
  - Control over the trade-off between accuracy and discrimination through the parameter :math:`\theta`.

- **Step-by-step description of the approach**

  1. **Single Classifier:**
     
     Consider a single probabilistic classifier that provides the posterior probability :math:`p(C^+|X)` for an instance :math:`X`. When :math:`p(C^+|X)` is close to 1 or 0, the classification decision is made with high certainty. However, when :math:`p(C^+|X)` is closer to 0.5, the decision is more uncertain. The ROC method introduces a reject option for instances where :math:`\max[p(C^+|X), 1 - p(C^+|X)] \leq \theta`, where :math:`0.5 < \theta < 1`. This defines the critical region.

     Instances in the critical region are relabeled as follows:

     - If the instance belongs to the deprived group :math:`\mathcal{X}_d`, it is labeled as :math:`C^+`.
     - If the instance belongs to the favored group :math:`\mathcal{X}_f`, it is labeled as :math:`C^-`.

     Instances outside the critical region are classified using the standard decision rule: :math:`C_i = \arg\max \{p(C^+|X_i), p(C^-|X_i)\}`.

  2. **Multiple Classifiers:**
     
     When using an ensemble of probabilistic classifiers, the posterior probability for an instance :math:`X` is computed as the weighted average of the posterior probabilities from each classifier in the ensemble. The weights can be proportional to the accuracy of each classifier or uniform if no prior information is available. The ROC method then proceeds as described for a single classifier, using the ensemble's posterior probability to define the critical region and relabel instances accordingly.

Basic Usage
~~~~~~~~~~~~~~
You can find an example of using the Reject Option based Classification (ROC) module in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/postprocessing.html#5.-Reject-Option>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.RejectOptionClassification`.

References
~~~~~~~~~~~~~~
1. Kamiran, Faisal, Asim Karim, and Xiangliang Zhang. "Decision theory for discrimination-aware classification." 2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.
