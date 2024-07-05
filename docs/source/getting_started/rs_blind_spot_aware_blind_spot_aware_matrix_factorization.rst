**Blind Spot Aware Matrix Factorization**
=========================================

**Introduction**
----------------
Blind Spot Aware Matrix Factorization (BSAMF) is a novel approach designed to address the limitations of conventional matrix factorization methods in recommender systems. The primary goal of BSAMF is to reduce the "blind spot" for users, which refers to the set of items that are predicted to have low ratings and thus are less likely to be recommended. By minimizing the blind spot, BSAMF aims to enhance the diversity of recommendations and improve user satisfaction.

**Description**
---------------
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

**Equations/Algorithms**
------------------------
The blind spot for user :math:`u` is defined as:

.. math::
    :label: equation-blind-spot

    D_u^\epsilon = \{i \in I \mid \hat{R}_{u,i} < \max_{u,i}(\hat{R}_{u,i}) \cdot \epsilon\}

where :math:`\epsilon` is a cut-off controlling the threshold.

The cost function for Blind Spot Aware Matrix Factorization is:

.. math::
    :label: equation-cost-function

    J = \sum_{u,i \in R} \|r_{u,i} - V_u^T M_i\|^2 + \frac{\lambda}{2} (\|V_u\|^2 + \|M_i\|^2) + \frac{\beta}{2} \|V_u - M_i\|^2

To minimize the objective function, stochastic gradient descent is used with the following updates:

.. math::
    :label: equation-update-V

    V_u \leftarrow V_u + \eta (2 e_{ui} M_i - \lambda V_u - \beta (V_u - M_i))

.. math::
    :label: equation-update-M

    M_i \leftarrow M_i + \eta (2 e_{ui} V_u - \lambda M_i + \beta (V_u - M_i))

where :math:`e_{ui} = \hat{r}_{ui} - V_u^T M_i`, :math:`\hat{r}_{ui}` is the predicted rating, and :math:`\eta` is the learning rate.

**Usage Examples**
------------------
The Blind Spot Aware Matrix Factorization method was tested on a synthetic dataset generated using item response theory. The dataset consisted of 500 users and 500 items, resulting in a total of 250,000 ratings. The method was evaluated based on the Root Mean Square Error (RMSE) and the Gini coefficient of item popularity.

**Advantages and Limitations**
------------------------------
*Advantages:*

- Reduces the blind spot size, leading to more diverse recommendations.
- Maintains a low RMSE, indicating accurate predictions.
- Decreases popularity bias compared to conventional matrix factorization.

*Limitations:*

- Assumes users always agree with the recommendations, which may not be realistic.
- Requires careful tuning of the blind spot penalty term weight coefficient :math:`\beta`.

**References**
---------------
1. Sun, Wenlong, et al. "Debiasing the human-recommender system feedback loop in collaborative filtering." Companion Proceedings of The 2019 World Wide Web Conference. 2019.