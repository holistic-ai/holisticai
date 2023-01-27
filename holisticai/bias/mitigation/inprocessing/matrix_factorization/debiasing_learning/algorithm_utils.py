import sys

import numpy as np


def process_propensities(observed_ratings, inverse_propensities):
    numObservations = np.ma.count(observed_ratings)
    numUsers, numItems = np.shape(observed_ratings)
    scale = numUsers * numItems
    inversePropensities = None
    if inverse_propensities is None:
        inversePropensities = (
            np.ones((numUsers, numItems), dtype=np.longdouble) * scale / numObservations
        )
    else:
        inversePropensities = np.array(
            inverse_propensities, dtype=np.longdouble, copy=True
        )

    inversePropensities = np.ma.array(
        inversePropensities,
        dtype=np.longdouble,
        copy=False,
        mask=np.ma.getmask(observed_ratings),
        fill_value=0,
        hard_mask=True,
    )
    return inversePropensities


def predict_scores(
    user_vectors, item_vectors, user_biases, item_biases, global_bias, use_bias=True
):
    rawScores = np.dot(user_vectors, item_vectors.T)
    if use_bias:
        biasedScores = (
            rawScores + user_biases[:, None] + item_biases[None, :] + global_bias
        )
        return biasedScores
    else:
        return rawScores


def normalize_propensities(
    inversePropensities, normalization, scale, numItems, numUsers
):
    perUserNormalizer = np.ma.sum(inversePropensities, axis=1, dtype=np.longdouble)
    perUserNormalizer = np.ma.masked_less_equal(perUserNormalizer, 0.0, copy=False)

    perItemNormalizer = np.ma.sum(inversePropensities, axis=0, dtype=np.longdouble)
    perItemNormalizer = np.ma.masked_less_equal(perItemNormalizer, 0.0, copy=False)

    globalNormalizer = np.ma.sum(inversePropensities, dtype=np.longdouble)

    normalizedPropensities = None
    if normalization == "Vanilla":
        normalizedPropensities = inversePropensities
    elif normalization == "SelfNormalized":
        normalizedPropensities = scale * np.ma.divide(
            inversePropensities, globalNormalizer
        )
    elif normalization == "UserNormalized":
        normalizedPropensities = numItems * np.ma.divide(
            inversePropensities, perUserNormalizer[:, None]
        )
    elif normalization == "ItemNormalized":
        normalizedPropensities = numUsers * np.ma.divide(
            inversePropensities, perItemNormalizer[None, :]
        )
    else:
        print("MF.GENERATE_MATRIX: [ERR]\t Normalization not supported:", normalization)
        sys.exit(0)
    return normalizedPropensities


def get_bias_configuration(bias_mode):
    useBias = None
    regularizeBias = None
    if bias_mode == "None":
        useBias = False
        regularizeBias = False
    elif bias_mode == "Regularized":
        useBias = True
        regularizeBias = True
    elif bias_mode == "Free":
        useBias = True
        regularizeBias = False
    else:
        print("MF.GENERATE_MATRIX: [ERR]\t Bias mode not supported:", bias_mode)
        sys.exit(0)
    return useBias, regularizeBias


def get_metric_mode(mode):
    metricMode = None
    if mode == "mse":
        metricMode = 1
    elif mode == "mae":
        metricMode = 2
    else:
        print("MF.GENERATE_MATRIX: [ERR]\t Metric not supported:", mode)
        sys.exit(0)
    return metricMode


def get_loss(delta, metricMode):
    if metricMode == 1:
        loss = np.square(delta)
    elif metricMode == 2:
        loss = np.abs(delta)
    else:
        sys.exit(0)
    return loss


def get_gradient_multiplier(metricMode, normalizedPropensities, delta):
    gradientMultiplier = None
    if metricMode == 1:
        gradientMultiplier = np.multiply(normalizedPropensities, 2 * delta)
    elif metricMode == 2:
        gradientMultiplier = np.zeros(np.shape(delta), dtype=np.int)
        gradientMultiplier[delta > 0] = 1
        gradientMultiplier[delta < 0] = -1
        gradientMultiplier = np.multiply(normalizedPropensities, gradientMultiplier)
    else:
        sys.exit(0)
    return gradientMultiplier


class ParameterSerializer:
    def __init__(self, numUsers, numItems, num_dimensions):
        self.numUsers = numUsers
        self.numItems = numItems
        self.num_dimensions = num_dimensions

    def serialize(
        self, user_vectors, item_vectors, user_biases, item_biases, global_bias
    ):
        allUserParams = np.concatenate((user_vectors, user_biases[:, None]), axis=1)
        allItemParams = np.concatenate((item_vectors, item_biases[:, None]), axis=1)

        allParams = np.concatenate((allUserParams, allItemParams), axis=0)
        paramVector = np.reshape(
            allParams, (self.numUsers + self.numItems) * (self.num_dimensions + 1)
        )
        paramVector = np.concatenate((paramVector, [global_bias]))
        return paramVector.astype(np.float)

    def deserialize(self, paramVector):
        globalBias = paramVector[-1]
        remainingParams = paramVector[:-1]
        allParams = np.reshape(
            remainingParams, (self.numUsers + self.numItems, self.num_dimensions + 1)
        )
        allUserParams = allParams[0 : self.numUsers, :]
        allItemParams = allParams[self.numUsers :, :]

        userVectors = (allUserParams[:, 0:-1]).astype(np.longdouble)
        userBiases = (allUserParams[:, -1]).astype(np.longdouble)

        itemVectors = (allItemParams[:, 0:-1]).astype(np.longdouble)
        itemBiases = (allItemParams[:, -1]).astype(np.longdouble)
        return userVectors, itemVectors, userBiases, itemBiases, globalBias


class Problem:
    def __init__(self, observed_ratings, normalized_propensities, config):
        self.observed_ratings = observed_ratings
        self.normalized_propensities = normalized_propensities
        self.serializer = config["serializer"]
        self.config = config

    def evaluate(self, paramVector):
        (
            userVectors,
            itemVectors,
            userBiases,
            itemBiases,
            globalBias,
        ) = self.serializer.deserialize(paramVector)
        biasedScores = predict_scores(
            userVectors,
            itemVectors,
            userBiases,
            itemBiases,
            globalBias,
            self.config["useBias"],
        )

        delta = np.subtract(biasedScores, self.observed_ratings)
        loss = get_loss(delta, self.config["metricMode"])

        weightedLoss = loss * self.normalized_propensities
        objective = np.sum(weightedLoss)

        gradientMultiplier = get_gradient_multiplier(
            self.config["metricMode"], self.normalized_propensities, delta
        )

        userVGradient = np.dot(gradientMultiplier, itemVectors)
        itemVGradient = np.dot(gradientMultiplier.T, userVectors)

        userBGradient = None
        itemBGradient = None
        globalBGradient = None
        if self.config["useBias"]:
            userBGradient = np.sum(gradientMultiplier, axis=1)
            itemBGradient = np.sum(gradientMultiplier, axis=0)
            globalBGradient = np.sum(gradientMultiplier)
        else:
            userBGradient = np.zeros_like(userBiases)
            itemBGradient = np.zeros_like(itemBiases)
            globalBGradient = 0.0

        if self.config["l2_regularization"] > 0:
            scaledPenalty = (
                1.0
                * self.config["l2_regularization"]
                * self.config["scale"]
                / (self.config["numUsers"] + self.config["numItems"])
            )
            if self.config["regularizeBias"]:
                scaledPenalty /= self.config["num_dimensions"] + 1
            else:
                scaledPenalty /= self.config["num_dimensions"]

            userVGradient += 2 * scaledPenalty * userVectors
            itemVGradient += 2 * scaledPenalty * itemVectors

            objective += scaledPenalty * np.sum(np.square(userVectors))
            objective += scaledPenalty * np.sum(np.square(itemVectors))

            if self.config["regularizeBias"]:
                userBGradient += 2 * scaledPenalty * userBiases
                itemBGradient += 2 * scaledPenalty * itemBiases
                globalBGradient += 2 * scaledPenalty * globalBias
                objective += scaledPenalty * np.sum(np.square(userBiases))
                objective += scaledPenalty * np.sum(np.square(itemBiases))
                objective += scaledPenalty * globalBias * globalBias

        gradient = self.serializer.serialize(
            userVGradient, itemVGradient, userBGradient, itemBGradient, globalBGradient
        )
        return objective, gradient


def init_random_parameters(config, start_vec=None):
    userVectorsInit = None
    itemVectorsInit = None
    userBiasesInit = None
    itemBiasesInit = None
    globalBiasInit = None
    if start_vec is None:
        userVectorsInit = np.random.standard_normal(
            (config["numUsers"], config["num_dimensions"])
        )
        itemVectorsInit = np.random.standard_normal(
            (config["numItems"], config["num_dimensions"])
        )
        userBiasesInit = np.zeros(config["numUsers"], dtype=np.float)
        itemBiasesInit = np.zeros(config["numItems"], dtype=np.float)
        globalBiasInit = 0
    else:
        userVectorsInit = start_vec[0]
        itemVectorsInit = start_vec[1]
        userBiasesInit = start_vec[2]
        itemBiasesInit = start_vec[3]
        globalBiasInit = start_vec[4]

    return (
        userVectorsInit,
        itemVectorsInit,
        userBiasesInit,
        itemBiasesInit,
        globalBiasInit,
    )
