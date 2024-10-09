from __future__ import annotations
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from holisticai.utils._definitions import ConditionalImportances, Importances

def get_importance_oscillation_table(partial_dependencies, importances, top_n):
    ocilations = partial_dependence_oscilation(partial_dependencies, importances, top_n=top_n, aggregated=False)
    idf = importances.as_dataframe().set_index('Variable').iloc[:top_n]
    odf = pd.DataFrame(ocilations, index=partial_dependencies.feature_names[:top_n], columns=['Oscillation'])
    return pd.concat([idf, odf], axis=1).reset_index().rename({'index': 'Variable'}, axis=1).sort_values(by='Importance', ascending=False)

def get_osillations_from_individuals(individuals):
    indice_oscilacion_normalizados = []
    for c in individuals:
        derivada = np.diff(c)
        cambios_signo = np.sum(np.diff(np.sign(derivada)) != 0)
        indice_oscilacion_normalizados.append(cambios_signo / len(c))
    return indice_oscilacion_normalizados

def partial_dependence_oscilation(partial_dependencies, importances, top_n=10, aggregated=True):
    df = importances.as_dataframe().sort_values('Importance', ascending=False).iloc[:top_n].set_index('Variable')
    feature_names = list(df.index)
    #print(feature_names)
    oscilacion = []
    for feature_name in feature_names:
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=1, data_type='individual')
        indice_oscilacion_normalizados = get_osillations_from_individuals(individuals)
        score = np.mean(indice_oscilacion_normalizados)
        weight = df.at[feature_name, 'Importance']
        #oscilacion.append(weight*score)
        oscilacion.append(score)
    if aggregated:
        return np.mean(oscilacion)
    return oscilacion


class RankAlignment:
    name: str = "Rank Alignment"
    reference: float = 1.0

    def __call__(self, conditional_feature_importance: ConditionalImportances, feature_importance: Importances, alpha=0.8, aggregation=True):
        top_feature_names = feature_importance.top_alpha(alpha=alpha).feature_names
        similarities = []
        for group_name, cond_features in conditional_feature_importance:
            top_cond_feature_names = cond_features.top_alpha(alpha=alpha).feature_names
            top_cond_feature_names = set(top_cond_feature_names)
            similarities.append(len(set(top_feature_names).intersection(top_cond_feature_names)) / len(set(top_feature_names).union(top_cond_feature_names)))
        if aggregation:
            return np.mean(similarities)
        return similarities

def rank_alignment(conditional_feature_importance: ConditionalImportances, ranked_feature_importance: Importances, aggregation=True):
    metric = RankAlignment()
    return metric(conditional_feature_importance, ranked_feature_importance, aggregation=aggregation)



def feature_rank_stability(local_importances):
    """
    Calculates the rank stability (RSt) for a set of ranked features derived from local feature importances.

    Args:
        local_importances (np.array): A matrix of shape (M, d), where M is the number of samples 
                                      and d is the number of features. Each entry represents the importance of a feature in a sample.
    
    Returns:
        float: The rank stability (RSt) value between 0 and 1.
    """
    # Convert local importances to rankings (higher importance gets a lower rank)
    ranked_features = np.argsort(-local_importances, axis=1)  # Sort each row in descending order, giving ranks
    
    # Number of iterations
    n = len(ranked_features)
    
    # Flatten the ranked features list and get the unique features
    unique_features = list(set([feature for iteration in ranked_features for feature in iteration]))
    
    # Initialize a dictionary to store ranks for each feature
    feature_ranks = {feature: [] for feature in unique_features}
    
    # Populate ranks for each feature
    for iteration in ranked_features:
        for rank, feature in enumerate(iteration, start=1):
            feature_ranks[feature].append(rank)
    
    # Calculate rank stability for each feature
    def rank_stability(feature_rankings):
        if len(feature_rankings) == 1:
            return 1.0  # If a feature is ranked only once, its stability is maximal
        
        # Find the most frequent (consistent) rank
        rank_counter = Counter(feature_rankings)
        most_frequent_rank = rank_counter.most_common(1)[0][0]
        
        # Calculate actual deviation (sum of absolute differences from the most frequent rank)
        actual_deviation = sum(abs(rank - most_frequent_rank) for rank in feature_rankings)
        
        # Calculate maximum deviation
        max_deviation = len(feature_rankings) * (max(feature_rankings) - min(feature_rankings))
        
        if max_deviation == 0:
            return 1.0  # If there's no variation, stability is maximal
        
        # Calculate and return rank stability
        return 1 - (actual_deviation / max_deviation)
    
    # Calculate overall rank stability for the system
    overall_stability = np.mean([rank_stability(feature_ranks[feature]) for feature in unique_features])
    
    return overall_stability



def feature_importance_stability(feature_importances):
    """
    Calculate the stability of local feature importance (e.g., SHAP values).
    
    Parameters:
    - feature_importances (np.array): A matrix of shape (M, d), where M is the number of samples 
      and d is the number of features. Each entry represents the importance of a feature in a sample.

    Returns:
    - stability (float): The stability metric, bounded between 0 and 1.
    """
    M, d = feature_importances.shape  # M: number of samples, d: number of features
    
    # Mean importance for each feature
    mean_importances = np.mean(feature_importances, axis=0)
    
    # Variance of the importance for each feature
    var_importances = (M / (M - 1)) * np.var(feature_importances, axis=0, ddof=1)  # ddof=1 for sample variance
    
    # Calculate stability
    stability_sum = 0
    for j in range(d):
        if mean_importances[j] == 0 or mean_importances[j] == 1:
            stability_j = 0  # If mean is 0 or 1, it is perfectly stable for that feature.
        else:
            # Variance normalized by mean and (1 - mean)
            stability_j = var_importances[j] / (mean_importances[j] * (1 - mean_importances[j]))
        stability_sum += stability_j
    
    stability = 1 - (1 / d) * stability_sum
    
    # Ensure stability is within bounds
    return max(0, min(1, stability))