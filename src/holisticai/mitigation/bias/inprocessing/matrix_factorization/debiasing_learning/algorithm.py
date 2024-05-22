import numpy as np
import scipy

from .algorithm_utils import (
    ParameterSerializer,
    Problem,
    get_bias_configuration,
    get_metric_mode,
    init_random_parameters,
    normalize_propensities,
    predict_scores,
    process_propensities,
)


class DebiasingLearningAlgorithm:
    def __init__(
        self, K, normalization, lamda, metric, bias_mode, clip_val=-1, verbose=0
    ):
        self.normalization = normalization
        self.metric = metric
        self.K = K
        self.clip_val = clip_val
        self.lamda = lamda
        self.bias_mode = bias_mode
        self.verbose = verbose

    def train(self, train_observations, inv_propensities, start_vector=None):
        if start_vector is not None:
            assert start_vector[0].shape[1] == self.K
            assert start_vector[1].shape[1] == self.K

        tempInvPropensities = None
        if inv_propensities is not None:
            tempInvPropensities = (4.0 / 3.0) * inv_propensities
            if self.clip_val >= 0:
                tempInvPropensities = np.clip(
                    tempInvPropensities, a_min=0, a_max=self.clip_val
                )

        parameters = self.optimize_debiasing_parameters(
            train_observations, tempInvPropensities, self.lamda, start_vec=start_vector
        )
        return parameters

    def optimize_debiasing_parameters(
        self, observed_ratings, inverse_propensities, l2_regularization, start_vec=None
    ):

        metricMode = get_metric_mode(self.metric)
        numUsers, numItems = np.shape(observed_ratings)
        scale = numUsers * numItems
        useBias, regularizeBias = get_bias_configuration(self.bias_mode)
        serializer = ParameterSerializer(
            numUsers=numUsers, numItems=numItems, num_dimensions=self.K
        )
        config = {
            "numUsers": numUsers,
            "numItems": numItems,
            "scale": scale,
            "useBias": useBias,
            "regularizeBias": regularizeBias,
            "metricMode": metricMode,
            "num_dimensions": self.K,
            "l2_regularization": l2_regularization,
            "serializer": serializer,
        }

        inversePropensities = process_propensities(
            observed_ratings, inverse_propensities
        )
        normalized_propensities = normalize_propensities(
            inversePropensities, self.normalization, scale, numItems, numUsers
        )
        normalized_propensities = np.ma.filled(normalized_propensities, 0.0)
        observed_ratings = np.ma.filled(observed_ratings, 0)

        problem = Problem(
            observed_ratings=observed_ratings,
            normalized_propensities=normalized_propensities,
            config=config,
        )

        startparameters = init_random_parameters(config, start_vec=start_vec)
        startVector = serializer.serialize(*startparameters)

        ops = {
            "maxiter": 2000,
            "disp": False,
            "gtol": 1e-5,
            "ftol": 1e-5,
            "maxcor": 50,
            "disp": False,
        }

        result = scipy.optimize.minimize(
            fun=problem.evaluate,
            x0=startVector,
            method="L-BFGS-B",
            jac=True,
            tol=1e-5,
            options=ops,
        )

        return serializer.deserialize(result["x"])

    def predict(self, model_params, bias_mode):
        use_bias = True
        if bias_mode == "None":
            use_bias = False
        return predict_scores(*model_params, use_bias=use_bias)
