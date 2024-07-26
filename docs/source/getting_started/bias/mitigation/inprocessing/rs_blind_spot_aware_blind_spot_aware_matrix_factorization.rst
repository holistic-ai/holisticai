Blind Spot Aware Matrix Factorization
----------------------

.. note::
    **Learning tasks:** Recommendation systems.

Introduction
~~~~~~~~~~~~~~~~
Blind Spot Aware Matrix Factorization (BSAMF) is a novel approach designed to address the limitations of conventional matrix factorization methods in recommender systems. The primary goal of BSAMF is to reduce the "blind spot" for users, which refers to the set of items that are predicted to have low ratings and thus are less likely to be recommended. By minimizing the blind spot, BSAMF aims to enhance the diversity of recommendations and improve user satisfaction.

Description
~~~~~~~~~~~~~~~~
The Blind Spot Aware Matrix Factorization method modifies the conventional matrix factorization approach by incorporating a term that penalizes the size of the blind spot. The blind spot for a user is defined as the set of items with predicted ratings below a certain threshold. This threshold is user-specific and is set to a percentile cut-off of the predicted ratings.

The method can be described as follows:

1. **Problem Definition**: Given a user-item rating matrix, the goal is to predict the missing ratings while minimizing the blind spot for each user.
2. **Main Characteristics**:

   - Incorporates a blind spot penalty term in the cost function.
   - Uses stochastic gradient descent for optimization.
   - Adjusts the user and item latent factors to reduce the blind spot size.
3. **Step-by-Step Description**:

   - Define the blind spot for each user based on a threshold.
   - Modify the cost function to include a term that penalizes the blind spot.
   - Use stochastic gradient descent to update the user and item latent factors.

References
~~~~~~~~~~~~~~~~
1. Sun, Wenlong, et al. "Debiasing the human-recommender system feedback loop in collaborative filtering." Companion Proceedings of The 2019 World Wide Web Conference. 2019.