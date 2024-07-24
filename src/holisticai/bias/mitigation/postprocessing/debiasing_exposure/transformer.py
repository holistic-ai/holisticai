import numpy as np
import pandas as pd
from holisticai.bias.mitigation.postprocessing.debiasing_exposure.algorithm import DELTRAlgorithm
from holisticai.bias.mitigation.postprocessing.debiasing_exposure.algorithm_utils import Standarizer


class DebiasingExposure:
    """
    Disparate Exposure Learning to Rank (DELTR) incorporates a measure of performance and a measure\
    of disparate exposure into its loss function. Trains a linear model based on performance and\
    fairness for a protected group.

    Parameters
    ----------
    group_col : str
        Name of the column in data that contains protected attribute.

    query_col : str
        Name of the column in data that contains query ids (optional).

    doc_col : str
        List of name of the column in data that contains document ids (optional).

    score_col : str
        Name of the column in data that contains judgment values (optional).

    feature_cols :
        Name of the columns in data that contains feature values  (optional).

    gamma : float
        Gamma parameter for the cost calculation in the training phase (recommended to be around 1).

    number_of_iterations : int
        Number of iteration in gradient descent (optional).

    learning_rate : float
        Learning rate in gradient descent (optional).

    lambdaa : float
        Regularization constant (optional).

    init_var : float
        Range of values for initialization of weights (optional).

    standardize : bool
        Boolean indicating whether the data should be standardized or not (optional).

    verbose : int
        If > 0, print progress.

    References
    ---------
    .. [1] Zehlike, Meike, and Carlos Castillo. "Reducing disparate exposure in ranking: A learning to rank\
        approach." Proceedings of The Web Conference 2020. 2020.
    """

    def __init__(
        self,
        group_col: str,
        query_col="query_id",
        doc_col="doc_id",
        score_col="judgment",
        feature_cols=None,
        gamma: float = 1.0,
        number_of_iterations=3000,
        learning_rate=0.001,
        lambdaa=0.001,
        init_var=0.01,
        standardize=False,
        verbose=0,
    ):
        if feature_cols is None:
            feature_cols = []

        # check if mandatory parameters are present
        if group_col is None:
            raise ValueError("The name of column in data `group_col` must be initialized")
        if gamma is None:
            raise ValueError("The `gamma` parameter must be initialized")

        # initialize the protected_feature index to -1

        # assign mandatory parameters
        self.group_col = group_col
        self.query_col = query_col
        self.doc_col = doc_col
        self.score_col = score_col
        self.feature_cols = feature_cols
        self._gamma = gamma

        # assign optional parameters
        self._number_of_iterations = number_of_iterations
        self._learning_rate = learning_rate
        self._lambda = lambdaa
        self._init_var = init_var
        self._standardize = standardize
        self.verbose = verbose
        self.standarizer = Standarizer(group_col=group_col)
        self.algorithm = DELTRAlgorithm(
            self._gamma,
            self._number_of_iterations,
            self._learning_rate,
            self._lambda,
            self._init_var,
            verbose=verbose,
        )

    def _filter_invalid_examples(self, rankings):
        new_rankings = []
        for _, ranking in rankings.groupby(self.query_col):
            if (ranking[self.group_col].sum() > 0).any():
                new_rankings.append(ranking)
        new_rankings = pd.concat(new_rankings, axis=0).reset_index(drop=True)
        return new_rankings

    def fit(self, rankings: pd.DataFrame):
        """
        Trains a Disparate Exposure model on a given training set.

        Parameters
        ----------
        rankings:  DataFrame

        Returns
        ------
            Self
        """

        rankings = self._filter_invalid_examples(rankings)
        if self.feature_cols == []:
            restricted_cols = [self.query_col, self.doc_col, self.score_col]
            self.feature_cols = [col for col in rankings.columns.to_list() if col not in restricted_cols]

        # prepare data
        (
            query_ids,
            doc_ids,
            protected_feature,
            feature_matrix,
            training_scores,
        ) = self._prepare_data(rankings, has_judgment=True)

        # standardize data if allowed
        if self._standardize:
            feature_matrix = self.standarizer.fit_transform(feature_matrix)

        self._omega = self.algorithm.fit(query_ids, protected_feature, training_scores, feature_matrix)

        # return model
        return self

    def transform(self, rankings: pd.DataFrame):
        """
        Train a Disparate Exposure model to rank the prediction set.

        Parameters
        ----------
        rankings:    DataFrame

        Returns
        ------
        DataFrame
            Transformed data
        """

        if self._omega is None:
            raise SystemError("You need to train a model first!")

        # prepare data
        query_ids, doc_ids, protected_attributes, feature_matrix = self._prepare_data(rankings, has_judgment=False)

        # standardize data if allowed
        if self._standardize:
            feature_matrix = self.standarizer.transform(feature_matrix)

        # calculate the predictions
        predictions = np.dot(feature_matrix, self._omega)

        # create the resulting data frame
        result = pd.DataFrame(
            {
                self.query_col: query_ids,
                self.doc_col: doc_ids,
                self.group_col: protected_attributes,
                self.score_col: predictions,
            }
        )

        # sort by the score in descending order
        result = result.sort_values([self.score_col], ascending=[0])

        return result

    def _prepare_data(self, data, has_judgment=False):
        """
        Extracts the different columns of the input data.

        Parameters
        ----------
        data: DataFrame

        has_adjudment: bool

        Return
        ------
        tuple
            Tuple of preprocessed data
        """
        query_ids = data[self.query_col]
        doc_ids = data[self.doc_col]
        protected_attributes = data[self.group_col]  # add 2 for query id and doc id

        feature_matrix = data[self.feature_cols]
        if has_judgment:
            scores = data[self.score_col]
            return query_ids, doc_ids, protected_attributes, feature_matrix, scores
        return query_ids, doc_ids, protected_attributes, feature_matrix
