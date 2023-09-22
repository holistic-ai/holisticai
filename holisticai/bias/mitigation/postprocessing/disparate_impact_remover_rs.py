import numpy as np
import pandas as pd

# from holisticai.bias.mitigation.commons.disparate_impact_remover.algorithm import GroupRepairer
from holisticai.bias.mitigation.preprocessing.disparate_impact_remover import (
    DisparateImpactRemover,
)
from holisticai.utils.transformers.bias import BMPostprocessing as BMPos
from holisticai.utils.transformers.bias import SensitiveGroups


class DisparateImpactRemoverRS(BMPos):
    """
    Disparate impact remover edits feature values to increase group fairness
    while preserving rank-ordering within groups.

    References
    ----------
        [1] Feldman, Michael, et al. "Certifying and removing disparate impact."
        proceedings of the 21th ACM SIGKDD international conference on knowledge
        discovery and data mining. 2015.
    """

    def __init__(
        self,
        query_col="X",
        group_col="protected",
        score_col="score",
        repair_level=1.0,
        verbose=0,
    ):
        """
        Description
        -----------
        Init Disparate Exposure in Learning To Rank class.

        Parameters
        ----------
        group_col: str
            Name of the column in data that contains protected attribute.

        score_col:
            Name of the column in data that contains judgment values.

        repair_level : float
            Repair amount 0.0 (min) -> 1.0 (max)

        verbose : int
            If > 0, print progress.
        """
        # assign mandatory parameters
        self.score_col = score_col
        self.query_col = query_col
        self.group_col = group_col
        self.verbose = verbose
        self._assert_parameters(repair_level)
        self.repair_level = repair_level
        self.sensgroup = SensitiveGroups()
        self.dir = DisparateImpactRemover(repair_level=repair_level)

    def _assert_parameters(self, repair_level):
        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")

    def _filter_invalid_examples(self, rankings):
        new_rankings = []
        for q, ranking in rankings.groupby(self.query_col):
            if (ranking[self.group_col].sum() > 0).any():
                new_rankings.append(ranking)
        new_rankings = pd.concat(new_rankings, axis=0).reset_index(drop=True)
        return new_rankings

    def fit_transform(self, rankings: pd.DataFrame):
        return self.fit().transform(rankings)

    def fit(self):
        return self

    def transform(self, rankings: pd.DataFrame):
        """
        Description
        -----------
        Train a Disparate Exposure model to rank the prediction set.

        Parameters
        ----------
        rankings:    DataFrame

        Return
        ------
            DataFrame
        """
        rankings = self._filter_invalid_examples(rankings)
        new_rankings = [
            self.transform_features(ranking)
            for q, ranking in rankings.groupby(self.query_col)
        ]

        return pd.concat(new_rankings, axis=0).reset_index(drop=True)

    def transform_features(self, ranking: pd.DataFrame):
        """
        Transform data

        Description
        -----------
        Transform data to a fair representation

        Parameters
        ----------
        ranking : DataFrame
            Input data

        Return
        ------
            Self
        """
        data = np.c_[ranking[self.score_col].to_numpy()].tolist()
        group_a = ranking[self.group_col] == True
        group_b = ranking[self.group_col] == False

        new_data_matrix = self.dir.fit_transform(data, group_a, group_b)
        new_ranking = ranking.copy()
        new_ranking[self.score_col] = new_data_matrix
        new_ranking = new_ranking.reset_index(drop=True)
        new_ranking = (
            new_ranking.groupby(self.query_col)
            .apply(lambda df: df.sort_values(self.score_col, ascending=False))
            .reset_index(drop=True)
        )
        return new_ranking
