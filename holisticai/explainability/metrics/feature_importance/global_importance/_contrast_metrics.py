import numpy as np
import pandas as pd

def __importance_range_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_feature_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    m_range = []
    for top_k in range(1, len(feature_importance_indexes) + 1):
        ggg = set(feature_importance_indexes[:top_k])
        vvv = set(conditional_feature_importance_indexes[:top_k])
        u = len(set(ggg).intersection(vvv)) / top_k
        m_range.append(u)
    m_range = np.array(m_range)

    return m_range.mean()


def __importance_order_constrast(
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
    m_order = np.array(feature_importance_indexes) == np.array(
        conditional_features_importance_indexes
    )
    m_order = np.cumsum(m_order) / np.arange(1, len(m_order) + 1)

    return m_order.mean()

def __important_similarity(feature_importance_indexes_1: np.ndarray, feature_importance_indexes_2: np.ndarray):
    from sklearn.metrics.pairwise import cosine_similarity
        
    f1 = np.array(feature_importance_indexes_1.sort_index()['Importance']).reshape([1,-1])
    f2 = np.array(feature_importance_indexes_2.sort_index()['Importance']).reshape([1,-1])
    
    return cosine_similarity(f1,f2)[0][0]

def important_constrast_matrix(cfimp, fimp, keys, show_connections=False):
    
    def nodes_and_edges(cfimp, fimp, keys, compare_fn, similarity=False):
        total_values = 2*len(keys)-1
        values = np.zeros(shape=(1,total_values))
        xticks = ['|']*total_values
        for i in range(1,len(keys)):
            if similarity:
                values[0,2*i-1] = compare_fn(cfimp[keys[i-1]], cfimp[keys[i]])
            else:
                values[0,2*i-1] = compare_fn(cfimp[keys[i-1]]['Variable'], cfimp[keys[i]]['Variable'])
                
        for i in range(len(keys)):
            if similarity:
                values[0,2*i] = compare_fn(fimp, cfimp[keys[i]])
            else:
                values[0,2*i] = compare_fn(fimp['Variable'], cfimp[keys[i]]['Variable'])
            xticks[2*i] = keys[i]
        return xticks , values

    def nodes_only(cfimp, fimp, keys, compare_fn, similarity=False):
        total_values = len(keys)
        values = np.zeros(shape=(1,total_values))
        xticks = ['|']*total_values
        for i in range(len(keys)):
            if similarity:
                values[0,i] = compare_fn(fimp, cfimp[keys[i]])
            else:
                values[0,i] = compare_fn(fimp['Variable'], cfimp[keys[i]]['Variable'])
            xticks[i] = keys[i]
        return xticks , values

    if show_connections:
        compare_importances_fn = nodes_and_edges
    else:
        compare_importances_fn = nodes_only
        
    xticks , range_values = compare_importances_fn(cfimp, fimp, keys, __importance_range_constrast)
    _ , order_values = compare_importances_fn(cfimp, fimp, keys, __importance_order_constrast)
    _ , sim_values = compare_importances_fn(cfimp, fimp, keys, __important_similarity, similarity=True)
    values = np.concatenate([order_values, range_values, sim_values], axis=0)
    return xticks,values

def feature_importance_contrast(
    feature_importance, conditional_feature_importance, mode=None, detailed=False
):

    feature_importance_indexes = list(feature_importance.index)
    conditional_feature_importance_indexes = {
        k: list(v.index) for k, v in conditional_feature_importance.items()
    }

    if mode == "range":
        feature_importance_constrast = {
            f"Global Range Overlap Score {k}": __importance_range_constrast(
                feature_importance_indexes, v
            )
            for k, v in conditional_feature_importance_indexes.items()
        }
        
        if not detailed:
            feature_importance_constrast = {'Global Range Overlap Score': np.mean(list(feature_importance_constrast.values()))}

    elif mode == "overlap":
        feature_importance_constrast = {
            f"Global Overlap Score {k}": __importance_order_constrast(
                feature_importance_indexes, v
            )
            for k, v in conditional_feature_importance_indexes.items()
        }
        if not detailed:
            feature_importance_constrast = {'Global Overlap Score': np.mean(list(feature_importance_constrast.values()))}
        
    elif mode == "similarity":
        feature_importance_constrast = {
            f"Global Similarity Score {k}": __important_similarity(
                feature_importance, v
            )
            for k, v in conditional_feature_importance.items()
        }
        if not detailed:
            feature_importance_constrast = {'Global Similarity Score': np.mean(list(feature_importance_constrast.values()))}

    return pd.DataFrame.from_dict(
        feature_importance_constrast, orient="index", columns=["Value"]
    )

