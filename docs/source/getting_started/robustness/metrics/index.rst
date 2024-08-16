==============
Metrics
==============

.. contents:: Table of Contents
   :local:
   :depth: 1


In machine learning, evaluating the robustness of models against adversarial attacks is critical for ensuring their reliability, particularly in security-sensitive applications. Two popular metrics used to measure this robustness are adversarial accuracy and empirical robustness. 

Adversarial accuracy
====================

Use case
---------

This metric evaluates the robustness of a machine learning model against adversarial attacks by considering the consistency of predictions and the preservation of correct classifications. In other words, it measures a model's ability to correctly classify inputs that an adversarial attack has intentionally perturbed. 

Given:

- :math:`x_i` are the original input samples.
- :math:`x_i^{\text{adv}}` are the adversarially perturbed samples.
- :math:`y_i` is the true label for sample :math:`x_i`.
- :math:`\hat{y}_i^{\text{orig}}` is the predicted label for the original sample :math:`x_i`.
- :math:`\hat{y}_i^{\text{adv}}` is the predicted label for the adversarial sample :math:`x_i^{\text{adv}}`.

The formula for adversarial accuracy for :math:`n` samples in this case is:

.. math::

   \text{Adversarial Accuracy} = \frac{\sum_{i=1}^{n} \left(\hat{y}_i^{\text{orig}} = \hat{y}_i^{\text{adv}} \text{ and } \hat{y}_i^{\text{orig}} = y_i \right)}{\sum_{i=1}^{n} \left(\hat{y}_i^{\text{orig}} = y_i \right)}

Interpretation 
--------------

Since measures the proportion of correctly classified samples under normal conditions that remain correctly classified (i.e., consistent predictions) under adversarial conditions. A higher value indicates better robustness, meaning the model retains its correctness despite adversarial perturbations.

Empirical robustness
====================

Use case
---------

This metric measures the robustness of different machine learning models or defenses against adversarial attacks by considering the relative size of perturbations required to fool the model. 
This calculation is done by quantifying the minimum perturbation necessary to cause a model to misclassify an input and evaluating how much an adversarial attack must alter the original input to fool the model successfully.

Given:

- :math:`x_i` are the original input samples.
- :math:`x_i^{\text{adv}}` are the adversarially perturbed samples.
- :math:`\hat{y}_i` is the predicted label for :math:`x_i`.
- :math:`\hat{y}_i^{\text{adv}}` is the predicted label for :math:`x_i^{\text{adv}}`.
- :math:`\|\cdot\|_p` is the :math:`p`-norm used to measure the distance (perturbation size).

The empirical robustness is:

.. math::

   \text{Empirical Robustness} = \frac{1}{|\{i : \hat{y}_i \neq \hat{y}_i^{\text{adv}}\}|} \sum_{i : \hat{y}_i \neq \hat{y}_i^{\text{adv}}} \frac{\| x_i^{\text{adv}} - x_i \|_p}{\| x_i \|_p}

where:

- :math:`\|x_i^{\text{adv}} - x_i\|_p` is the norm of the perturbation applied to the input :math:`x_i`.
- :math:`\|x_i\|_p` is the norm of the original input :math:`x_i`.
- The summation is taken over all samples :math:`i` where the adversarial attack was successful (i.e., :math:`\hat{y}_i \neq \hat{y}_i^{\text{adv}}`).

**Observation**: A value of 0 indicates that the predictions are equal for the original and adversarial samples.

Interpretation 
--------------

Since this metric measures the mean ratio of the adversarial perturbations to the norm of the original inputs for those samples misclassified by the adversarial attack, more significant perturbations are needed to change the classifier's prediction. Therefore, a higher robustness score indicates greater difficulty for attackers in fooling the model.

References
~~~~~~~~~~
1. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2574-2582).