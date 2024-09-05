HopSkipJumpAttack
-----------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~

HopSkipJumpAttack is an iterative algorithm designed to generate adversarial examples against machine learning models. Its significance lies in its ability to function effectively even on models lacking gradients, such as those employing non-differentiable functions or discontinuous input transformations. This expands the applicability of adversarial attack techniques to a broader range of model architectures.

Description
~~~~~~~~~~~

**Problem definition:** 

The goal is to generate adversarial examples that are imperceptible perturbations of input data points but cause a target machine learning model to misclassify them. This problem arises in security and robustness testing of machine learning systems, where understanding vulnerabilities to malicious manipulations is crucial. 

**Main features:** 

HopSkipJumpAttack offers the following key advantages:

- **Gradient-free:** It operates without requiring access to the gradients of the target model, making it applicable to a wider range of models including non-differentiable ones.
- **Query-efficient:** The algorithm minimizes the number of queries required to the target model, improving efficiency compared to some other decision-based attacks.

**Step-by-step description of the approach**: 

1.  The attack begins with an initial input data point :math:`x_0`.

2. It then iteratively perturbs :math:`x_0` in a direction estimated to be towards the decision boundary, aiming for misclassification by the model.
3. The perturbation direction is determined using an unbiased estimate of the gradient direction at the decision boundary based on model predictions. This estimation technique forms a core contribution of HopSkipJumpAttack.

4. The magnitude of the perturbation is carefully controlled to ensure that the resulting adversarial example remains close to the original input and thus appears imperceptible.
5. The process continues until the perturbed input successfully fools the target model into misclassifying it, generating an effective adversarial example.

References
~~~~~~~~~~~
1. Chen, J., Jordan, M. I., & Wainwright, M. J. (2020, May). Hopskipjumpattack: A query-efficient decision-based attack. In 2020 ieee symposium on security and privacy (sp) (pp. 1277-1294). IEEE.