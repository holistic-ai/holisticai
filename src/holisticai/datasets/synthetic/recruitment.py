from __future__ import annotations

import random

import pandas as pd


def generate_rankings(
    n_rank: int | None,
    k: int | None,
    p: float | None,
    return_p_attr=False,  # noqa: FBT002
):
    """
    Generates M rankings of k elements (candidates) using Yang-Stoyanovich process

    Parameters
    ----------
    n_rank: int
        Number of rankings to generate
    k: int
        Number of elements should each ranking have
    p: float
        Probability that a candidate is protected

    Return
    ------
        DataFrame or Tuple of DataFrames
    """
    rankings = []
    for n in range(n_rank):
        for i in range(k):
            is_protected = random.random() <= p  # noqa: S311
            rankings.append({"X": n, "Y": k - i, "score": k - i, "protected": is_protected})

    df = pd.DataFrame(rankings)
    if return_p_attr:
        return df[["X", "Y", "score"]], df[["X", "Y", "protected"]]
    return df[["X", "Y", "score", "protected"]]
