import numpy as np
from holisticai.bias.mitigation.inprocessing.matrix_factorization.common_utils.utils import calculate_popularity_model


def constant_propensity(rmat, numItems):
    numObservations = np.ma.count(rmat)
    numUsers, numItems = np.shape(rmat)
    scale = numUsers * numItems
    inversePropensities = np.ones((numUsers, numItems), dtype=np.longdouble) * scale / numObservations
    return inversePropensities


def popularity_model_propensity(rmat):
    prop_score = calculate_popularity_model(rmat)
    IversPropensity = [prop_score] * rmat.shape[0]
    return np.array(IversPropensity)[:, 0, :]
