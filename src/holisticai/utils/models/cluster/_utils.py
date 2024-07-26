import numpy as np


def distance(a, b, order=2):
    """
    Description
    -----------
    Calculates the specified norm between two vectors.

    Parameters
    ----------
        a : list
            First vector
        b : list
            Second vector:
        order :  int
            Order of the norm to be calculated as distance

    Returns
    -------
        Resultant norm value
    """
    assert len(a) == len(b), "Length of the vectors for distance don't match."
    return np.linalg.norm(x=np.array(a) - np.array(b), ord=order)
