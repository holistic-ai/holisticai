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

   The permutation feature importance measures the importance of a feature by calculating the increase in the model's prediction error after the feature's values have been perturbed. In this context, the perturbation involves shuffling the feature's values.

   With this approach, a feature is considered **more important** if shuffling its values leads to a significant increase in the model's prediction error. Conversely, a feature is considered **less important** if shuffling its values results in little to no change in the model's prediction error.

   The basic algorithm for permutation feature importance is:

      Input: trained model :math:`\hat{f}`, feature matrix :math:`X`, target :math:`y`, error measure :math:`\mathcal{L}(y, \hat{f})`

      1. Estimate model error :math:`\epsilon_{0}=\mathcal{L}(y, \hat{f}(X))`
      2. For each feature :math:`j` in :math:`1, \dots, p`:
         - generate a permuted feature matrix :math:`\bar{X}`
         - estimate :math:`\bar{\epsilon}=\mathcal{L}(y, \bar{X})`
         - compute the permutation feature importance  ratio :math:`\mathcal{F}_{j}=\frac{\bar{\epsilon}}{\epsilon_{0}}` or the difference :math:`\mathcal{F}_{j}=\bar{\epsilon}-\epsilon_{0}`
      3. Sort the features by descending :math:`\mathcal{F}`

   This algorithm follow the implementation proposed by `Fisher et al. (2018) <https://arxiv.org/abs/1801.01489>`_.
  
- **Surrogate Models**: Surrogate models involve approximating the complex model with a simpler, interpretable model (like a decision tree). By fitting the surrogate model to the predictions of the complex model, we can gain insights into the decision-making process of the original model.

   A surrogate model is an interpretable model designed to approximate the predictions of a more complex machine learning model. The goal is to achieve a model that balances good accuracy with interpretability. To obtain a surrogate model we can employ the following steps:

      Input: dataset :math:`X`, a black-box model :math:`g`, a interpretable model :math:`f` 

      1. Select a dataset :math:`X`
      2. Get the predictions of :math:`g`
      3. Train :math:`f` on :math:`X` and get its predictions
      4. Measure the performance of :math:`f` to replicate the predictions of :math:`g` (e.g. R-squared)
      5. Interpret the results of surrogate model


Local Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~~

Local feature importance focuses on understanding the contribution of features to individual predictions. Two popular methods for local feature importance are SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

- **SHAP**: provide a unified measure of feature importance for individual predictions by computing the contribution of each feature to the prediction. This method is based on cooperative game theory and ensures fair attribution of feature importance.

   The `SHAP <https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_ presents a unified framework for interpreting model's predictions. SHAP assign each feature an importance value for a particular prediction. The work also show that game theory results guaranteeing a unique solution apply to additive feature attribution methods and SHAP is a solution with some desired proprieties: (1) local accuracy, (2) missingness, and (3) consistency.

   Additive feature attribution have an explanation model :math:`g` that is linear function of binary variables 

   .. math::
      g(z') = \phi_{0} + \sum\limits_{i=1}^{M}\phi_{i}z'_{i}

   where :math:`z'\in \{0,1\}^{M}`, :math:`M` is the number of simplified input features, and :math:`\phi_{i}\in\mathbb{R}`.

   Methods with the previous equation attribute an effect :math:`\phi_{i}` to each feature, and
   summing the effects of all feature attributions approximates the output :math:`f(x)` of the original model. There are several methods that match this definition, like SHAP and LIME. But the paper argues that SHAP is the unique model that follows the equation and satisfies the desired proprieties 1, 2 and 3.

   To achieved this, SHAP uses Shapley values, a result observed in cooperative game theory. We can define Shapley values as 

   .. math::
      \phi_{i}(f, x) = \sum_{z'\subseteq x'} \frac{|z'|!(M-|z'|-1)!}{M!}[f_{x}(z')-f_{x}(z'_{i})]

   where :math:`|z'|` is non-zero entries in :math:`z'`, and :math:`z'\subseteq x'` represents all :math:`z'` vectors where the non-zero entries are subset of non-zero entries in :math:`x'`.
  
- **LIME**: explains individual predictions by approximating the complex model with an interpretable model locally around the prediction. By perturbing the input data and observing the changes in predictions, LIME identifies the most influential features for that specific instance.

   The main goal of `LIME <https://arxiv.org/pdf/1602.04938>`_ is propose an explanation method to be applied in any classifier or regression model. In this context, explain is presenting visual or texts artifacts that provides qualitative understanding of the relationship between the instances components and the model's predictions. 

   We can define the explanations produced by LIME as 

   .. math::
      \mathcal{E}(x) = arg\min\limits_{g\in G} ~\mathcal{L}(f, g, \pi_{x})+\Omega (g)

   So, the explanation :math:`\mathcal{E}` of a given instance $x$ is equal the minimization of the fidelity function :math:`\mathcal{L}` while having the complexity :math:`\Omega (g)` low enough to be interpretable by humans. In this way, :math:`\mathcal{L}(f,g, \pi_{x})` measure how unfaithful the model $g$ is in approximating the model :math:`f` being explained in the locality defined by :math:`\pi_{x}`.

   The results find in the paper indicate that LIME is useful to increase trust in black-box models and model selection (avoiding models with good accuracy but with wrong motivations, i.e, using *a priori* "non-sense" features to make predictions). 


==============

Measuring and Mitigation
------------------------

.. toctree::
    :maxdepth: 2

    metrics