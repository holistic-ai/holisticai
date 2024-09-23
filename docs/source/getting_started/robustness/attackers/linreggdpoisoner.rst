Linear Gradient-Based Poisoning Attack
--------------------------------------

.. note::
    **Learning tasks:** Regression.

Introduction
~~~~~~~~~~~~~~~
The linear gradient-based poisoning attack aims to inject malicious data points into a training dataset for linear regression models, thereby manipulating the model's learned parameters and degrading its predictive performance. This attack leverages the gradients of the loss function with respect to the input features to strategically position poisoned points that maximize their impact on the regression line.

Description
~~~~~~~~~~~~~~

**Problem definition**
The goal is to introduce a small number of carefully crafted data points into the training set such that they significantly alter the learned regression line, leading to inaccurate predictions on unseen data. 


**Main features**

- **Gradient-based optimization:** The attack utilizes gradient information from the loss function to guide the placement of poisoned points.

- **Strategic point selection:** Poisoned points are chosen based on their ability to maximize the change in the regression line's slope and intercept, effectively "pulling" it away from its optimal position.
    
**Step-by-step description of the approach**

1.  **Initialization**: Start with a set of clean training data :math:`{(x_i, y_i)}` for :math:`i = 1,...,n`.

2. **Gradient Calculation:** Compute the gradient of the loss function (e.g., mean squared error) with respect to each feature dimension in the input space. This gradient indicates the direction in which changes to the features will have the greatest impact on the model's predictions.
3.  **Poison Point Selection**: Select a point :math:`x_p` in the input space based on the calculated gradients. The goal is to choose a point that maximizes its distance from the current regression line, as this will induce the largest shift in the line's parameters.

4. **Response Value Assignment:** Assign a response value :math:`y_p` to the poison point :math:`x_p`. This value should be chosen strategically to further amplify the impact on the regression line (e.g., selecting a value that is significantly different from the predicted value for :math:`x_p` based on the clean model).

5. **Iteration**: Repeat steps 2-4 until a desired number of poison points have been generated and added to the training dataset.

Basic Usage
~~~~~~~~~~~~~~

Read more about the class attributes and methods in the API reference: :class:`~holisticai.robustness.attackers.LinRegGDPoisoner`.

References
~~~~~~~~~~~~~~
1. Jagielski, M., Oprea, A., Biggio, B., Liu, C., Nita-Rotaru, C., & Li, B. (2018, May). Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In 2018 IEEE symposium on security and privacy (SP) (pp. 19-35). IEEE.