import matplotlib.pyplot as plt
import numpy as np


def distance(a, b, order=2):
    """
    Calculates the specified norm between two vectors.

    Args:
            a (list) : First vector
            b (list) : Second vector
            order (int) : Order of the norm to be calculated as distance

    Returns:
            Resultant norm value
    """
    assert len(a) == len(b), "Length of the vectors for distance don't match."
    return np.linalg.norm(x=np.array(a) - np.array(b), ord=order)


def balance_calculation(data, centers, mapping):
    """
    Checks fairness for each of the clusters defined by k-centers.
    Returns balance using the total and class counts.

    Args:
            data (list)
            centers (list)
            mapping (list) : tuples of the form (data, center)

    Returns:
            fair (dict) : key=center, value=(sum of 1's corresponding to fairness variable, number of points in center)
    """
    fair = dict([(i, [0, 0]) for i in centers])
    for i in mapping:
        fair[i[1]][1] += 1
        if data[i[0]][0] == 1:  # MARITAL
            fair[i[1]][0] += 1

    curr_b = []
    for i in list(fair.keys()):
        p = fair[i][0]
        q = fair[i][1] - fair[i][0]
        if p == 0 or q == 0:
            balance = 0
        else:
            balance = min(float(p / q), float(q / p))
        curr_b.append(balance)

    return min(curr_b)


def plot_analysis(degrees, costs, balances, step_size):
    """
    Plots the curves for costs and balances.

    Args:
            degrees (list)
            costs (list)
            balances (list)
            step_size (int)
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].plot(costs, marker=".", color="blue")
    ax[0].set_xticks(list(range(0, len(degrees), step_size)))
    ax[0].set_xticklabels(
        list(range(min(degrees), max(degrees) + 1, step_size)), fontsize=12
    )
    ax[1].plot(balances, marker="x", color="saddlebrown")
    ax[1].set_xticks(list(range(0, len(degrees), step_size)))
    ax[1].set_xticklabels(
        list(range(min(degrees), max(degrees) + 1, step_size)), fontsize=12
    )
    plt.show()
