import numpy as np
import pandas as pd


def importance_range_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_features_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    feature_importance_indexes = list(feature_importance_indexes)
    conditional_features_importance_indexes = list(conditional_features_importance_indexes)
    min_len = min(len(feature_importance_indexes), len(conditional_features_importance_indexes))
    feature_importance_indexes = feature_importance_indexes[:min_len]
    conditional_features_importance_indexes = conditional_features_importance_indexes[:min_len]
    m_range = []
    for top_k in range(1, len(feature_importance_indexes) + 1):
        ggg = set(feature_importance_indexes[:top_k])
        vvv = set(conditional_features_importance_indexes[:top_k])
        u = len(set(ggg).intersection(vvv)) / top_k
        m_range.append(u)
    m_range = np.array(m_range)

    return m_range.mean()


def importance_order_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_features_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    feature_importance_indexes = list(feature_importance_indexes)
    conditional_features_importance_indexes = list(conditional_features_importance_indexes)
    min_len = min(len(feature_importance_indexes), len(conditional_features_importance_indexes))
    feature_importance_indexes = feature_importance_indexes[:min_len]
    conditional_features_importance_indexes = conditional_features_importance_indexes[:min_len]
    m_order = [
        f == c
        for f, c in zip(
            feature_importance_indexes, conditional_features_importance_indexes
        )
    ]
    m_order = np.cumsum(m_order) / np.arange(1, len(m_order) + 1)

    return m_order.mean()


def important_similarity(
    feature_importance_indexes_1: np.ndarray, feature_importance_indexes_2: np.ndarray
):
    from sklearn.metrics.pairwise import cosine_similarity

    f1 = np.array(feature_importance_indexes_1["Importance"]).reshape([1, -1])
    f2 = np.array(feature_importance_indexes_2["Importance"]).reshape([1, -1])

    return cosine_similarity(f1, f2)[0][0]


def important_constrast_matrix(acfimp, afimp, cfimp, fimp, keys, show_connections=False):
    def nodes_and_edges(cfimp, fimp, keys, compare_fn, similarity=False):
        total_values = 2 * len(keys) - 1
        values = np.zeros(shape=(1, total_values))
        xticks = ["|"] * total_values
        for i in range(1, len(keys)):
            if similarity:
                values[0, 2 * i - 1] = compare_fn(cfimp[keys[i - 1]], cfimp[keys[i]])
            else:
                values[0, 2 * i - 1] = compare_fn(
                    cfimp[keys[i - 1]].index, cfimp[keys[i]].index
                )

        for i in range(len(keys)):
            if similarity:
                values[0, 2 * i] = compare_fn(fimp, cfimp[keys[i]])
            else:
                values[0, 2 * i] = compare_fn(fimp.index, cfimp[keys[i]].index)
            xticks[2 * i] = keys[i]
        return xticks, values

    def nodes_only(cfimp, fimp, keys, compare_fn, similarity=False):
        total_values = len(keys)
        values = np.zeros(shape=(1, total_values))
        xticks = ["|"] * total_values
        for i in range(len(keys)):
            if similarity:
                values[0, i] = compare_fn(fimp, cfimp[keys[i]])
            else:
                values[0, i] = compare_fn(fimp.index, cfimp[keys[i]].index)
            xticks[i] = keys[i]
        return xticks, values

    if show_connections:
        compare_importances_fn = nodes_and_edges
    else:
        compare_importances_fn = nodes_only

    xticks, range_values = compare_importances_fn(
        cfimp, afimp, keys, importance_range_constrast
    )
    _, order_values = compare_importances_fn(
        cfimp, afimp, keys, importance_order_constrast
    )
    _, sim_values = compare_importances_fn(
        cfimp, fimp, keys, important_similarity, similarity=True
    )
    values = np.concatenate([order_values, range_values, sim_values], axis=0)
    return xticks, values


class ContrastMetric:
    def __init__(self, detailed, contrast_function):
        self.detailed = detailed
        self.contrast_function = contrast_function

    def __call__(self, feat_imp, cond_feat_imp):

        cond_contrast = {
            f"{self.name} {k}": self.contrast_function(feat_imp, cfi)
            for k, cfi in cond_feat_imp.items()
        }
        contrast = {self.name: np.mean(list(cond_contrast.values()))}

        if self.detailed:
            return {**contrast, **cond_contrast}

        return contrast


class PositionParity(ContrastMetric):
    def __init__(self, detailed):
        self.reference = 1
        self.name = "Position Parity"
        contrast_fn = lambda x, y: importance_order_constrast(x.index, y.index)
        super().__init__(detailed, contrast_fn)


class RankAlignment(ContrastMetric):
    def __init__(self, detailed):
        self.reference = 1
        self.name = "Rank Alignment"
        contrast_fn = lambda x, y: importance_range_constrast(x.index, y.index)
        super().__init__(detailed, contrast_fn)


class ImportantSimilarity(ContrastMetric):
    def __init__(self, detailed):
        self.reference = 1
        self.name = "Important Similarity"
        super().__init__(detailed, important_similarity)
