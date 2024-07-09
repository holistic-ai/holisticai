==============
Explainability
==============

Despite the remarkable recent evolution in prediction performance by artificial intelligence (AI) models, they are often deemed as "black boxes", i.e., models whose prediction mechanisms cannot be understood simply from their parameters. Explainability in machine learning refers to the ability to understand and articulate how models arrive at their predictions. This is crucial for promoting transparency, trust, and accountability in AI systems. It helps in verifying model behavior, refining models, debugging unexpected behavior, and communicating model decisions to stakeholders.

.. contents:: Table of Contents
   :local:
   :depth: 1


Feature Importance
------------------

Feature importance is a key approach to analyzing explainability. It assesses the contribution of each feature to the model's predictions. There are two main types of feature importance: global and local.

Global Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~~~

Global feature importance provides insights into the overall model by indicating how much each feature contributes to the model's predictions across the entire dataset. Two common methods for global feature importance are permutation and surrogate models.

- **Permutation Feature Importance**: This method involves shuffling the values of each feature and measuring the change in the model's error. If shuffling a feature's values increases the error significantly, the feature is considered important. This approach is model-agnostic and intuitive.
  
- **Surrogate Models**: Surrogate models involve approximating the complex model with a simpler, interpretable model (like a decision tree). By fitting the surrogate model to the predictions of the complex model, we can gain insights into the decision-making process of the original model.

Local Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~~

Local feature importance focuses on understanding the contribution of features to individual predictions. Two popular methods for local feature importance are SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

- **SHAP (SHapley Additive exPlanations)**: SHAP values provide a unified measure of feature importance for individual predictions by computing the contribution of each feature to the prediction. This method is based on cooperative game theory and ensures fair attribution of feature importance.
  
- **LIME (Local Interpretable Model-agnostic Explanations)**: LIME explains individual predictions by approximating the complex model with an interpretable model locally around the prediction. By perturbing the input data and observing the changes in predictions, LIME identifies the most influential features for that specific instance.


==============

Measuring and Mitigation
------------------------

.. toctree::
    :maxdepth: 2

    metrics