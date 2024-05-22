import numpy as np
from tqdm import trange

from .algorithm_utils import (
    exposure_diff,
    find_items_per_group_per_query,
    hh,
    normalized_topp_prot_deriv_per_group,
    topp,
)


class DELTRAlgorithm(object):
    def __init__(
        self, gamma, number_of_iterations, learning_rate, lambdaa, init_var, verbose
    ):
        # assign parameters
        self._gamma = gamma
        self._number_of_iterations = number_of_iterations
        self._learning_rate = learning_rate
        self._lambda = lambdaa
        self._init_var = init_var
        self.verbose = verbose

        self._no_exposure = False
        if gamma == 0:
            self._no_exposure = True

        self.log = []

        self._data_per_query = {}

    def fit(self, query_ids, protected_feature, training_scores, feature_matrix):
        """
        Description
        -----------
        Train a Neural Network to find the optimal feature weights in listwise learning to rank.

        Parameters
        ----------
        query_ids: list
            List of query IDs.

        feature_matrix: matrix-like
            training features.

        training_scores: array-like
            training judgments.

        Return
        ------
            model parameters.
        """
        self.indexes_per_query = {
            q: np.where(np.array(query_ids == q))[0] for q in query_ids.unique()
        }
        self.queries = list(self.indexes_per_query.keys())

        n_features = feature_matrix.shape[1]
        for q, indexes in self.indexes_per_query.items():
            self._data_per_query[
                hh(q, training_scores)
            ] = find_items_per_group_per_query(
                training_scores.iloc[indexes], protected_feature.iloc[indexes]
            )

            self._data_per_query[
                hh(q, feature_matrix)
            ] = find_items_per_group_per_query(
                feature_matrix.iloc[indexes], protected_feature.iloc[indexes]
            )
        self.feature_data_per_query_list = []
        self.true_data_per_query_list = []
        self.prot_idx_per_query_list = []
        for query in self.queries:
            feature_data_per_query = self._data_per_query[hh(query, feature_matrix)][0]
            true_data_per_query = self._data_per_query[hh(query, training_scores)][0]
            prot_idx_per_query = protected_feature.iloc[self.indexes_per_query[query]]

            self.feature_data_per_query_list.append(feature_data_per_query)
            self.true_data_per_query_list.append(true_data_per_query)
            self.prot_idx_per_query_list.append(prot_idx_per_query)

        # linear neural network parameter initialization
        omega = (np.random.rand(n_features, 1) * self._init_var).reshape(-1)
        for t in trange(self._number_of_iterations, leave=self.verbose > 0):
            # forward propagation
            predicted_scores = np.dot(feature_matrix, omega)
            predicted_scores = np.reshape(
                predicted_scores, (feature_matrix.shape[0], 1)
            )
            grad = self._calculate_gradient(predicted_scores)
            omega = omega - self._learning_rate * np.sum(
                np.asarray(grad), axis=0
            ).reshape(-1)

        return omega

    def _calculate_cost(
        self, training_judgments, predictions, prot_idx, data_per_query_predicted
    ):
        """
        Description
        -----------
        Computes the loss in list-wise learning to rank.

        Parameters
        ----------
        training_judgments: pd.Series
            containing the training judgments/ scores.

        predictions: pd.Series
            containing the predicted scores.

        prot_idx: pd.Series
            list stating which item is protected or non-protected.

        data_per_query_predicted:
            stores all judgments and all predicted scores that belong to one query.

        Return
        ------
            float value --> loss
        """
        results = [
            self._loss(
                query,
                training_judgments,
                predictions,
                prot_idx,
                data_per_query_predicted,
            )
            for query in self.queries
        ]

        # calucalte losses for better debugging
        loss_standard = sum(results)[0]
        loss_exposure = sum(
            [
                exposure_diff(predictions[indexes], prot_idx[indexes])
                for indexes in self.indexes_per_query.values()
            ]
        )

        return np.asarray(results), loss_standard, loss_exposure

    def _loss(
        self,
        which_query,
        training_judgments,
        predictions,
        prot_idx,
        data_per_query_predicted,
    ):
        """Calculate loss for a given query"""

        result = -np.dot(
            np.transpose(
                topp(self._data_per_query[(hh(which_query, training_judgments))][0])
            ),
            np.log(topp(data_per_query_predicted[hh(which_query, predictions)][0])),
        ) / np.log(predictions.size)

        if not self._no_exposure:
            indexes = self.indexes_per_query[which_query]
            result += (
                self._gamma
                * exposure_diff(predictions[indexes], prot_idx[indexes]) ** 2
            )

        return result

    def _calculate_gradient(self, predictions):
        """
        Description
        -----------
        calculates local gradients of current feature weights.

        Parameters
        ----------
        predictions:
            Vector containing the prediction scores.

        Return
        ------
            float value --> optimal listwise cost.
        """
        # Pool
        # from multiprocessing import Pool
        # with Pool(4) as p:
        #    results = list(p.starmap(self._grad, [(query, training_features, training_judgments, predictions, prot_idx,
        #                      data_per_query_predicted) for query in self.queries]))

        # Joblib
        # results = Parallel(n_jobs=4, verbose=0)(
        #   delayed(self._grad)(query, training_features, training_judgments, predictions, prot_idx,
        #                      data_per_query_predicted) for query in self.queries
        # )

        predictions_per_query_list = []
        for query, prot_idx_per_query in zip(
            self.queries, self.prot_idx_per_query_list
        ):
            predictions_per_query = predictions[self.indexes_per_query[query]].astype(
                "float32"
            )
            predictions_per_query = np.exp(
                find_items_per_group_per_query(
                    predictions_per_query, prot_idx_per_query
                )[0]
            )
            predictions_per_query_list.append(predictions_per_query)

        args_list = (
            self.feature_data_per_query_list,
            predictions_per_query_list,
            self.prot_idx_per_query_list,
            self.true_data_per_query_list,
        )
        results = [self._grad(predictions.size, *args) for args in zip(*args_list)]
        return np.asarray(results)

    def _grad(
        self,
        predictions_size,
        feature_data_per_query,
        predictions_per_query,
        prot_idx_per_query,
        true_data_per_query,
    ):

        feature_data_per_query_transpose = np.transpose(feature_data_per_query)

        result = 1 / np.sum(predictions_per_query)
        # l3
        result *= np.dot(feature_data_per_query_transpose, predictions_per_query)
        # l1
        result += -np.dot(feature_data_per_query_transpose, topp(true_data_per_query))
        # L deriv
        result /= np.log(predictions_size)

        if not self._no_exposure:
            result = result.reshape(-1) + self._gamma * 2 * exposure_diff(
                predictions_per_query, prot_idx_per_query
            ) * self._normalized_topp_prot_deriv_per_group_diff(
                feature_data_per_query, predictions_per_query, prot_idx_per_query
            )
        return result

    def _normalized_topp_prot_deriv_per_group_diff(
        self, training_features_per_query, predictions_per_query, prot_idx_per_query
    ):
        """
        Description
        -----------
        Calculates the difference of the normalized topp_prot derivative of the protected and non-protected groups.

        Parameters
        ----------
        training_features:
            vector of all features.

        predictions:
            predictions of all data points

        prot_idx:
            list stating which item is protected or non-protected.

        Return
        ------
            numpy array of float values.
        """

        (
            train_judgments_per_query,
            train_protected_items_per_query,
            train_nonprotected_items_per_query,
        ) = find_items_per_group_per_query(
            training_features_per_query, prot_idx_per_query
        )

        (
            predictions_per_query,
            pred_protected_items_per_query,
            pred_nonprotected_items_per_query,
        ) = find_items_per_group_per_query(predictions_per_query, prot_idx_per_query)

        u2 = normalized_topp_prot_deriv_per_group(
            train_nonprotected_items_per_query,
            train_judgments_per_query,
            pred_nonprotected_items_per_query,
            predictions_per_query,
        )  # derivative for non-protected group
        u3 = normalized_topp_prot_deriv_per_group(
            train_protected_items_per_query,
            train_judgments_per_query,
            pred_protected_items_per_query,
            predictions_per_query,
        )  # derivative for protected group

        return u2 - u3

    def losses(self):
        return self.log
