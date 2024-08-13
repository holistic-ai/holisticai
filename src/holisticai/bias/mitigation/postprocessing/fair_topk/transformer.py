from __future__ import annotations

from typing import Optional

import pandas as pd
from holisticai.bias.mitigation.postprocessing.fair_topk.algorithm_utils.fail_prob import (
    RecursiveNumericFailProbabilityCalculator,
)
from holisticai.bias.mitigation.postprocessing.fair_topk.algorithm_utils.valitation_utils import (
    check_ranking,
    validate_basic_parameters,
)
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost


class FairTopK(BMPost):
    """
    Fair Top K bias mitigation [1]_ can be used for Recommender Systems.\
    The strategy extends group fairness definition using the standard notion of protected groups\
    and is based on ensuring that the proportion of protected candidates in every prefix of the top-k\
    ranking.

    Parameters
    ----------
    top_n : int
        The total number of elements.

    p : float
        The proportion of protected candidates in the top-k ranking.

    alpha : float
        The significance level.

    query_col : str
        The name of the column in data that contains query ids.

    doc_col : str
        The name of the column in data that contains document ids.

    group_col : str
        The name of the column in data that contains protected attribute.

    score_col : str
        The name of the column in data that contains judgment values.

    References
    ---------
    .. [1] Zehlike, Meike, et al. "Fa* ir: A fair top-k ranking algorithm." Proceedings of the 2017 ACM on\
    Conference on Information and Knowledge Management. 2017.
    """

    def __init__(
        self,
        top_n: Optional[int],
        p: Optional[float],
        alpha: Optional[float],
        query_col: Optional[str] = "query_id",
        doc_col: Optional[str] = "doc_id",
        group_col: Optional[str] = "group_id",
        score_col: Optional[str] = "score",
    ):
        # check the parameters first
        validate_basic_parameters(top_n, p, alpha)
        self.query_col = query_col
        self.doc_col = doc_col
        self.group_col = group_col
        self.score_col = score_col

        # assign the parameters
        self.top_n = top_n  # the total number of elements
        self.p = p  # the proportion of protected candidates in the top-k ranking
        self.alpha = alpha  # the significance level
        self._cache = {}  # stores generated mtables in memory

    def transform(self, rankings, p_attr=None):
        """
        Apply transform to prediction scores.

        Parameters
        ----------
        rankings : DataFrame
            Predicted matrix scores (nb_examples*top_n, 3) [query_id, doc_id, scores]

        p_attr: matrix-like
            Item groups (nb_examples, 3) [query_id, doc_id, protected]

        Returns
        -------
        DataFrame
            The re-ranked dataframe.
        """
        if p_attr is None:
            if self.group_col not in rankings.columns:
                raise ValueError("protected groups must be provided")
            new_rankings = rankings
        else:
            if self.group_col in rankings.columns:
                del rankings[self.group_col]

            new_rankings = pd.merge(rankings, p_attr, on=[self.query_col, self.doc_col], how="left")

        query_result_by_group = new_rankings.groupby(self.query_col)
        re_rankings = [df if self.is_fair(df) else self.transform_ranking(df) for _, df in query_result_by_group]
        return pd.concat(re_rankings).reset_index(drop=True)

    def transform_ranking(self, ranking):
        """
        Applies FA*IR re-ranking to the input ranking using an adjusted mtable

        Parameters
        ----------
        ranking: list
            The ranking to be re-ranked (list of FairScoreDoc)

        Returns
        ------
        DataFrame
            The re-ranked dataframe.
        """
        protected = ranking[ranking[self.group_col]]
        non_protected = ranking[~ranking[self.group_col]]
        mtable = self._create_adjusted_mtable()
        return pd.DataFrame(self._fair_top_k(protected, non_protected, mtable)).reset_index(drop=True)

    def _create_adjusted_mtable(self):
        """
        Description
        -----------
        Creates an adjusted mtable by using the alpha value.

        Return
        ------
        list
            mtable as list of int elements
        """

        if (self.top_n, self.p, self.alpha) not in self._cache:
            # create the mtable
            fail_prob_pair = RecursiveNumericFailProbabilityCalculator(self.top_n, self.p, self.alpha).adjust_alpha()
            mtable = [int(i) for i in fail_prob_pair.mtable.m.tolist()]
            # store as list
            self._cache[(self.top_n, self.p, self.alpha)] = mtable

        # return from cache
        return self._cache[(self.top_n, self.p, self.alpha)]

    def is_fair(self, ranking):
        """
        Checks if the ranking is fair for the given parameters

        Parameters
        ----------
        ranking: list
            The ranking to be checked (list of Resultinfo)

        Returns
        ------
        bool
            True if the ranking is fair, False otherwise.
        """
        return check_ranking(ranking[self.group_col], self._create_adjusted_mtable())

    def _fair_top_k(self, protected_candidates, non_protected_candidates, mtable):
        """
        Reorganize the results info ensuring true the mtable condition (#protected[:i] >= mtable[i]).

        Parameters
        ----------

        protected_candidates:  pd.DataFrame
            ranking dataframe filtered with only protected candidates

        non_protected_candidates:  pd.DataFrame
            ranking dataframe filtered with only non protected candidates

        mtable: list
            adjusted mtable

        Returns
        ------
        list
            List of re-ranked results.
        """
        result = []
        countProtected = 0

        idxProtected = 0
        idxNonProtected = 0

        for i in range(self.top_n):
            if idxProtected >= len(protected_candidates) and idxNonProtected >= len(non_protected_candidates):
                # no more candidates available, return list shorter than k
                return result
            if idxProtected >= len(protected_candidates):
                # no more protected candidates available, take non-protected instead
                result.append(non_protected_candidates.iloc[idxNonProtected])
                idxNonProtected += 1

            elif idxNonProtected >= len(non_protected_candidates):
                # no more non-protected candidates available, take protected instead
                result.append(protected_candidates.iloc[idxProtected])
                idxProtected += 1
                countProtected += 1
            elif countProtected < mtable[i]:
                # add a protected candidate
                result.append(protected_candidates.iloc[idxProtected])
                idxProtected += 1
                countProtected += 1
            elif (
                protected_candidates.iloc[idxProtected][self.score_col]
                >= non_protected_candidates.iloc[idxNonProtected][self.score_col]
            ):
                # the best is a protected one
                result.append(protected_candidates.iloc[idxProtected])
                idxProtected += 1
                countProtected += 1
            else:
                # the best is a non-protected one
                result.append(non_protected_candidates.iloc[idxNonProtected])
                idxNonProtected += 1

        return result
