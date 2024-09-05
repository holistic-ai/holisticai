ZOO: Zeroth Order Optimization
-------------------------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~
Zeroth order optimization (ZOO) methods are derivative-free optimization techniques used to find optimal solutions for functions where calculating gradients directly is challenging or impossible. These methods rely solely on function evaluations, making them suitable for scenarios with black-box models where gradient information is unavailable. ZOO leverages the concept of approximating gradients using finite differences and then employs classical optimization algorithms like gradient descent or coordinate descent on these estimated gradients to converge towards a solution.

Description
~~~~~~~~~~~~

**Problem definition**

Zeroth order optimization (ZOO) aims to find the minimum (or maximum) value of an objective function :math:`f(x)` without access to its derivative information. This is particularly relevant in scenarios involving black-box models, where the internal workings are unknown and gradient calculations are infeasible.

**Main features**
    
The key features of ZOO methods include:

- **Derivative-free:** ZOO methods do not require calculating gradients explicitly. They rely on evaluating the objective function at different points.
- **Approximation of Gradients:**  Gradients are approximated using finite differences, this allows the use of classical optimization algorithms, to iteratively update the solution towards the solution.

**Step-by-step description of the approach**

1.  **Initialization:** Choose an initial point :math:`x_0` in the search space.
2. **Gradient Approximation:** For a given direction vector :math:`v`, estimate the gradient along that direction using finite differences:

    .. math::
        \nabla f(x) \approx \frac{f(x + hv) - f(x - hv)}{2h}

    where :math:`h` is a small step size and :math:`v` is a direction vector.

3.  **Optimization Step:** Apply a classical optimization algorithm, such as gradient descent or coordinate descent, to update the solution based on the approximated gradient.

4. **Iteration:** Repeat steps 2 and 3 until a convergence criterion is met and the adversarial sample is generated.

References
~~~~~~~~~~
1. Chen, P. Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C. J. (2017, November). Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security (pp. 15-26).