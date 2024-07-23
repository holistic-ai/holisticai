import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_ranking(protected, mtable):
    """
    Description
    -----------
    Checks if the ranking is fair in respect to the mtable

    Parameters
    ----------
    ranking: DataFrame
        The ranking to be checked.

    mtable: List
        The mtable against to check (list of int)

    Returns
    ------
        bool
        Returns whether the rankings satisfies the mtable
    """
    # if the mtable has a different number elements than there are in the top docs return false
    if len(protected) != len(mtable):
        raise ValueError("Number of documents in ranking and mtable length must be equal!")

    # check number of protected element at each rank
    return not (protected.cumsum().values < np.array(mtable)).any()


def validate_basic_parameters(k, p, alpha):
    """
    Description
    -----------
    Validates if k, p and alpha are in the required ranges

    Parameters
    ----------
        k : int
            Total number of elements (above or equal to 10)

        p : int
            The proportion of protected candidates in the top-k ranking (between 0.02 and 0.98)

        alpha : float
            The significance level (between 0.01 and 0.15)
    """
    if k < 10 or k > 400:
        if k < 2:
            raise ValueError("Total number of elements `k` should be between 10 and 400")
        logger.warning("Library has not been tested with values outside this range")

    if p < 0.02 or p > 0.98:
        if p < 0 or p > 1:
            raise ValueError(
                "The proportion of protected candidates `p` in the top-k ranking should be between " "0.02 and 0.98"
            )
        logger.warning("Library has not been tested with values outside this range")

    validate_alpha(alpha)


def validate_alpha(alpha):
    if alpha < 0.01 or alpha > 0.15:
        if alpha < 0.001 or alpha > 0.5:
            raise ValueError("The significance level `alpha` must be between 0.01 and 0.15")
        logger.warning("Library has not been tested with values outside this range")
