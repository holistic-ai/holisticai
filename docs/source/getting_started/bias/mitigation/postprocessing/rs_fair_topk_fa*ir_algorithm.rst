FA*IR Algorithm
-----------------

.. note::
    **Learning tasks:** Recommendation systems.

Introduction
~~~~~~~~~~~~~~~
The FA*IR algorithm is designed to address the problem of ranking candidates while ensuring fairness constraints are met. It aims to balance the utility of the ranking with the need to include a minimum proportion of protected candidates. This method is particularly relevant in contexts where fairness and diversity are critical, such as in hiring processes, academic admissions, or any selection process where bias mitigation is necessary.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  The FA*IR algorithm tackles the problem of ranking candidates (referred to as the Fair Top-k Ranking problem) while ensuring that a specified minimum proportion of protected candidates is included in the top-k positions. The algorithm takes into account the qualifications of each candidate and adjusts the ranking to meet fairness constraints.

- **Main features**

  The main features of the FA*IR algorithm include:
  - Creation of priority queues for protected and non-protected candidates based on their qualifications.
  - Calculation of a ranked group fairness table to determine the minimum number of protected candidates required at each position.
  - Greedy construction of the ranking to maximize utility while satisfying fairness constraints.
  - Efficient running time.

- **Step-by-step description of the approach**

  1. **Initialization**:

     - The algorithm initializes two empty priority queues, :math:`P_0` for non-protected candidates and :math:`P_1` for protected candidates, each with a bounded capacity of :math:`k`.
     - It then inserts each candidate into the appropriate priority queue based on their qualifications.

  2. **Ranked Group Fairness Table**:

     - For each position from 1 to :math:`k`, the algorithm computes the minimum number of protected candidates required using a function :math:`F^{-1}(\alpha_c; i, p)`, where :math:`\alpha_c` is the adjusted significance level, :math:`i` is the position, and :math:`p` is the target minimum proportion of protected candidates.

  3. **Greedy Construction of the Ranking**:

     - The algorithm initializes counters :math:`t_p` and :math:`t_n` to zero, representing the number of protected and non-protected candidates added to the ranking, respectively.
     - It iteratively constructs the ranking by checking the ranked group fairness table:
       - If the current position requires a protected candidate, the best candidate from :math:`P_1` is added to the ranking.
       - Otherwise, the best candidate from the combined pool of :math:`P_0` and :math:`P_1` is added, ensuring that the candidate with the highest qualification is selected.

  4. **Finalization**:

     - The algorithm continues this process until the ranking contains :math:`k` candidates.
     - It returns the final ranking, which satisfies the group fairness condition and maximizes utility.

The FA*IR algorithm ensures that the ranking is fair by construction, adhering to the in-group monotonicity and ranked group fairness constraints. It guarantees that the number of protected candidates in any prefix of the ranking meets the required minimum, thus balancing fairness and utility effectively.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Fair Top-k method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/recommender_systems/demos/postprocessing.html#Method:-Fair-Top-K>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.FairTopK`.

References
~~~~~~~~~~~~~~
1. Zehlike, Meike, et al. "Fa* ir: A fair top-k ranking algorithm." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
