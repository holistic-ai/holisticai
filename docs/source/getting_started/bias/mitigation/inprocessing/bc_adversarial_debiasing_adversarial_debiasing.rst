Adversarial Debiasing
----------------------

.. note::
    **Learning tasks:** Binary classification.


Introduction
~~~~~~~~~~~~~~~
Adversarial Debiasing is a method designed to mitigate unwanted biases in machine learning models by leveraging adversarial learning techniques. The method aims to ensure that the predictions made by a model are not influenced by protected variables, such as gender or race, which are considered sensitive and should not affect the decision-making process. This approach is significant because it addresses the ethical and fairness concerns in machine learning, ensuring that models do not perpetuate existing biases present in the training data.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  Machine learning models often inherit biases from the training data, leading to unfair predictions that can discriminate against certain demographic groups. The goal of Adversarial Debiasing is to train a model that accurately predicts an output variable :math:`Y` from an input variable :math:`X`, while ensuring that the predictions are unbiased with respect to a protected variable :math:`Z`. The protected variable :math:`Z` could be any sensitive attribute such as gender, race, or zip code.

- **Main features**

  The Adversarial Debiasing method incorporates an adversarial network into the training process. This adversarial network is designed to predict the protected variable :math:`Z` from the model's predictions :math:`\hat{Y}`. The main features of this method include:
  
  - Simultaneous training of a predictor and an adversary.
  - The predictor aims to maximize the accuracy of predicting :math:`Y`.
  - The adversary aims to minimize its ability to predict :math:`Z` from :math:`\hat{Y}`.
  - The method can be applied to various definitions of fairness, such as Demographic Parity and Equality of Odds.
  - It is flexible and can be used with different types of gradient-based learning models, including both regression and classification tasks.

- **Step-by-step description of the approach**

  1. **Predictor Training**: The primary model, referred to as the predictor, is trained to predict the output variable :math:`Y` from the input variable :math:`X`. The predictor's objective is to minimize the prediction loss :math:`L_P(\hat{y}, y)`, where :math:`\hat{y}` is the predicted value and :math:`y` is the true value.

  2. **Adversary Training**: An adversarial network is introduced, which takes the predictor's output :math:`\hat{Y}` as input and attempts to predict the protected variable :math:`Z`. The adversary's objective is to minimize its prediction loss :math:`L_A(\hat{z}, z)`, where :math:`\hat{z}` is the adversary's predicted value of :math:`Z` and :math:`z` is the true value of :math:`Z`.

  3. **Adversarial Objective**: The adversarial network's loss is incorporated into the predictor's training process. The predictor is trained not only to minimize its own prediction loss :math:`L_P`, but also to maximize the adversary's loss :math:`L_A`. This is achieved by updating the predictor's weights in a way that reduces the information about :math:`Z` contained in :math:`\hat{Y}`.

  4. **Fairness Constraints**: Depending on the desired fairness definition, the adversary's input may vary. For Demographic Parity, the adversary only receives :math:`\hat{Y}` as input. For Equality of Odds, the adversary also receives the true label :math:`Y` as input, ensuring that any information about :math:`Z` in :math:`\hat{Y}` is limited to what is already contained in :math:`Y`.

  5. **Training Process**: The training process involves alternating updates between the predictor and the adversary. The predictor is updated to improve its prediction accuracy while deceiving the adversary. The adversary is updated to improve its ability to predict :math:`Z` from :math:`\hat{Y}`. This adversarial training continues until a balance is achieved where the predictor makes accurate predictions of :math:`Y` without revealing information about :math:`Z`.

  6. **Evaluation**: The trained model is evaluated to ensure that it meets the desired fairness criteria. Metrics such as False Positive Rate (FPR) and False Negative Rate (FNR) are used to assess whether the model's predictions are unbiased with respect to the protected variable :math:`Z`.

References
~~~~~~~~~~~~~~
1. B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.