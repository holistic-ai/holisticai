from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import pandas as pd
from lime import lime_tabular

def get_index_groups(model_type, y):
    if model_type == 'binary_classification':
        index_groups = {f'[label={value}]':(y==value).index for value in y.unique()}
        return index_groups
    
    elif model_type == 'regression':
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(y.quantile(labels_values)).squeeze()
        index_groups = {f'[{c}]':y[(y.values > v[i]) & (y.values < v[i+1])].index for (i, c) in enumerate(labels)}
        return index_groups
    else:
        raise NotImplementedError

def lime_creator(scorer, X, index_groups=None, num_features=None, num_samples=None, mode='classification'):
    ##################
    # scorer - predict function or predict_proba (classification)
    # X - input data, such as model.predict(X) -> ypred | multi column pandas df
    # index_groups (old stratified) - list of lists, each sublist with booleans with the same lenght as X.index
    # num_features - number of features to compute local feature weight
    # num_samples - number of samples
    ###################

    # load and do assignment
    if num_features is None:
        num_features = np.min([X.shape[1], 50])

    if num_samples is None:
        num_samples = np.min([X.shape[0], 50])

    per_group_sample = int(np.ceil(num_samples / len(index_groups)))
    ids_groups = {str(label):np.random.choice(X.index[index], size=per_group_sample).tolist() for label,index in index_groups.items()}

    # calculate lime for several samples
    explainer = lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns.tolist(), discretize_continuous=True, mode=mode)

    df = []
    for label, indexes in ids_groups.items():
        for i in indexes:
            exp = explainer.explain_instance(X.loc[i], scorer, num_features=X.shape[1], num_samples=100)
            exp_values = list(exp.local_exp.values())[0]

            df_i = pd.DataFrame(exp_values, columns=["Feature Id", "Feature Weight"])
            df_i["Importance"] = df_i["Feature Weight"].abs()
            df_i["Importance"] = df_i["Importance"]/df_i["Importance"].max()
            df_i["Sample Id"] = i
            df_i["Feature Label"] = X.columns[df_i["Feature Id"].tolist()]
            df_i["Feature Rank"] = range(1, df_i.shape[0]+1)
            df_i['Sample Group'] = label
            df.append(df_i)

    df = pd.concat(df, axis=0, ignore_index=True)

    return df


def four_fifths_list_lime(feature_importance, feature_importance_names, cutoff=None):
    ######################
    # feature_importance - array with raw feature importance
    # feature_importance_names - list with names
    # cutoff - if None, use 0.8
    #####################

    # four-fifths or another cutoff point
    if cutoff is None:
        cutoff = 0.80

    # feature weight
    feature_weight = feature_importance/sum(feature_importance)

    # entropy or divergence
    return feature_importance_names.loc[(feature_weight.cumsum() < cutoff).values]

def four_fifths_list(feature_importance, cutoff=None):
    ######################
    # feature_importance - array with raw feature importance
    # feature_importance_names - list with names
    # cutoff - if None, use 0.8
    #####################

    # four-fifths or another cutoff point
    if cutoff is None:
        cutoff = 0.80

    importance = feature_importance['Importance']
    feature_names = feature_importance['Variable']
    # feature weight
    feature_weight = importance/sum(importance)

    # entropy or divergence
    return feature_names.loc[(feature_weight.cumsum() < cutoff).values]

def partial_dependence_creator(model, grid_resolution, x, feature_ids, target=None):
    ##################
    # model - sklearn-like object with predict function and predict_proba
    # grid_resolution - grid_resolution for pdp
    # X - input data, such as model.predict(X) -> ypred | multi column pandas df
    ###################

    # to do
    # -> explicit implementation of plot_partial_dependence_plot from sklearn without showing the chart
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['interactive'] == False
    # load and do assignment

    feature_names = list(x.columns)
    method='brute'
    percentiles=(0.05, 0.95)
    response_method='auto'
    # calculate partial dependence
    kargs = {
        'estimator':model,
        'X':x,
        'features':feature_ids,
        'feature_names':feature_names,
        'percentiles':percentiles,
        'response_method':response_method,
        'method':method,
        'grid_resolution':grid_resolution
    }

    if not target==None:
        kargs.update({'target':target})

    g = PartialDependenceDisplay.from_estimator(**kargs)
    plt.close()

    # store and export results
    pd_results = {}
    for (i, f) in enumerate(feature_ids):
        pd_results[feature_names[f]] = pd.DataFrame({"score": g.pd_results[i]["average"][0], "values": g.pd_results[i]["values"][0]})

    return pd_results

def importance_spread(feature_importance, divergence=False):
    ######################
    # feature_importance - array with raw feature importance
    # divergence - if True calculate divergence instead of ratio
    #####################
    if len(feature_importance)==0:
        return 0 if divergence else 1

    importance = feature_importance
    from scipy.stats import entropy
    feature_weight = importance/sum(importance)
    feature_equal_weight = np.array([1.0/len(importance)]*len(importance))

    # entropy or divergence
    if divergence is False:
        return entropy(feature_weight) / entropy(feature_equal_weight) # ratio
    else:
        return entropy(feature_weight, feature_equal_weight) # divergence

def explanation_contrast(feature_importance, cond_feature_importance, order=True):
    ######################
    # feature_importance - df with global feature importance
    # conditional_feature_importance - dict with dfs with feature importance per conditional
    # order - True in case we are measuring the contrast with respect to order, False if analysing weight
    #####################

    # todo - implement per weight

    # in case we are per measuring order
    if order:
        # conditionals
        conditionals = list(cond_feature_importance.keys())

        # contrast per conditional
        results = []
        for c in conditionals:
            matching = feature_importance["Variable"].index==cond_feature_importance[c]["Variable"].index
            results += (c, matching.mean())

    # entropy or divergence
    return results


def importance_range_constrast(feature_importance_indexes:np.ndarray, conditional_feature_importance_indexes:np.ndarray):
    m_range = []
    for top_k in range(1, len(feature_importance_indexes)+1):
        ggg = set(feature_importance_indexes[:top_k])
        vvv = set(conditional_feature_importance_indexes[:top_k])
        u = len(set(ggg).intersection(vvv))/top_k
        m_range.append(u)
    m_range = np.array(m_range)

    return m_range.mean()

def importance_order_constrast(feature_importance_indexes:np.ndarray, conditional_features_importance_indexes:np.ndarray):
    m_order = [feature_importance_indexes == conditional_features_importance_indexes]
    m_order = np.cumsum(m_order)/np.arange(1, len(m_order) + 1)

    return m_order

    