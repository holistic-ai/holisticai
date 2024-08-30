Fair Regression via Plug-In Estimator and Recalibration
-------------------------------------------------------

.. note::
    **Learning tasks:** Regression.

Introduction
~~~~~~~~~~~~
The method addresses the problem of learning an optimal regression function subject to a fairness constraint. This constraint ensures that, conditionally on the sensitive feature, the distribution of the function output remains the same. The method is computationally efficient and statistically principled, aiming to mitigate or remove unfairness in regression tasks.

Description
~~~~~~~~~~~
- **Problem definition**

  The goal is to design a regression function that not only minimizes prediction error but also satisfies a fairness constraint. Specifically, the fairness constraint requires that the distribution of the regression function's output is invariant with respect to the sensitive feature. This is a generalization of demographic parity to regression problems.

- **Main features**

  This method consists of two main stages:
  
  1. **Estimation of the unconstrained regression function**: This is done using standard regression techniques on a labeled dataset.
  2. **Recalibration of the regression function**: This step uses a separate set of unlabeled data to adjust the initial regression function to satisfy the fairness constraint. The recalibration is performed via a smooth optimization process.

- **Step-by-step description of the approach**

  1. **Initial Regression Function Estimation**:

     - Estimate the regression function :math:`\eta` using standard methods on the labeled data.

  3. **Recalibration**:

     - Define the final estimator :math:`\hat{g}_L`.

  4. **Optimization**:

     - The minimization problem in the recalibration step is convex and can be efficiently solved using a smoothing technique based on Nesterov's method.
     - The optimization algorithm iteratively updates the parameters to find an approximate solution that satisfies the fairness constraint.

Basic Usage
~~~~~~~~~~~~~~
You can find an example of using the Plugin estimator and calibrator Method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/regression/demos/postprocessing.html#Bias-Mitigation>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.PluginEstimationAndCalibration`.

References
~~~~~~~~~~~~~~
1. Chzhen, Evgenii, et al. "Fair regression via plug-in estimator and recalibration with statistical guarantees." Advances in Neural Information Processing Systems 33 (2020): 19137-19148.
