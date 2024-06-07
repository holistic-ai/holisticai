import random
from typing import Optional

import pandas as pd


def generate_rankings(
    M: Optional[int], k: Optional[int], p: Optional[float], return_p_attr=False, seed: Optional[int] = 42
):
    """
    Generates M rankings of k elements (candidates) using Yang-Stoyanovich process

    Parameters
    ----------
    M: int
        Number of rankings to generate
    k: int
        Number of elements should each ranking have
    p: float
        Probability that a candidate is protected
    return_p_attr: bool
        Whether to return protected attribute
    seed: int
        Random seed

    Returns
    -------
    DataFrame, Tuple
        DataFrame or Tuple of DataFrames
    """
    random.seed(seed)
    rankings = []
    for m in range(M):
        for i in range(k):
            is_protected = random.random() <= p
            rankings.append(
                {"X": m, "Y": k - i, "score": k - i, "protected": is_protected}
            )

    df = pd.DataFrame(rankings)
    if return_p_attr:
        return df[["X", "Y", "score"]], df[["X", "Y", "protected"]]
    return df[["X", "Y", "score", "protected"]]
