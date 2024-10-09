import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from holisticai.utils import concatenate_metrics

plt.style.use('ggplot') 
plt.rcParams.update({'font.size': 12}) 

def plot_metrics_vs_accuracy(model_metrics, xai_column):
    plt.figure()
    xdf = concatenate_metrics(model_metrics).T.iloc[:-1]
    plt.plot(xdf['Accuracy'], xdf[xai_column], '*')
    for _, row in xdf.iterrows():
        plt.text(row['Accuracy'], row[xai_column], row.name.replace("DecisionTreeClassifier","DT"), fontsize=8, color='gray')
        plt.xlabel('Accuracy')
        plt.ylabel(xai_column)

def plot_metrics_vs(model_metrics, table, metric1, metric2, metric_3):
    plt.figure(figsize=(15, 10))
    xdf = concatenate_metrics(model_metrics).T.iloc[:-1]
    normalized_metric = metric_3
    for _, row in xdf.iterrows():
        plt.plot(row[metric1], row[metric2], '*', markersize=20, color=plt.cm.viridis((row[normalized_metric] - xdf[normalized_metric].min()) / (xdf[normalized_metric].max() - xdf[normalized_metric].min())))
        short_model_name = row.name.replace("DecisionTreeClassifier", "DT")
        plt.text(row[metric1], row[metric2], short_model_name, fontsize=12, color='gray', alpha=0.3)
        #if row.name in set(table.index)-set(["DecisionTreeClassifier-1"]):
        #    plt.scatter(row[metric1], row[metric2], linewidths=1, edgecolor='red', facecolors='none', s=500)  # Ajustar el tamaño a 100 para hacer visibles los círculos huecos
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    norm = plt.Normalize(vmin=xdf[normalized_metric].min(), vmax=xdf[normalized_metric].max())
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=plt.gca())
    cbar.set_label(normalized_metric)
    plt.grid()

def plot_partial_dependence(partial_dependencies, top_n=None, subplots=None, figsize=None):
    if figsize is None:
        figsize = (20, 5)
    
    if top_n is None:
        top_n = min(4, len(partial_dependencies.feature_names))

    if subplots is None:
        subplots = ((top_n + 4 - 1)//4, 4)

    _, axs = plt.subplots(*subplots, figsize=figsize, dpi=200)
    color_map = cm.get_cmap('PuBu')

    for i, feature_name in enumerate(partial_dependencies.feature_names[:4]):
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='individual')
        average = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='average')
        grid_values = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='grid_values')
        
        num_curves = len(individuals)
        
        for idx, c in enumerate(individuals):
            axs[i].plot(grid_values, c, color=color_map(idx / num_curves), alpha=0.02)
        
        axs[i].plot(grid_values, average, color='r', linewidth=2)
        axs[i].grid()
        axs[i].set_xlabel(feature_name)
        axs[i].set_ylabel('Partial Dependence')
        axs[i].set_title("[feature={}]".format(feature_name))
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()


def plot_feature_importance_contrast(X, proxy, importances, conditional_feature_importances):
    feature_names = []
    for name, cimp in conditional_feature_importances.values.items():
        feature_names += cimp.top_alpha().feature_names
    feature_names = sorted(set(feature_names))
    
    # Prepare dataframes
    imp_df = importances.as_dataframe().set_index('Variable').rename({'Importance': "Overall"}, axis=1).loc[feature_names]
    cimp_df = [cimp.as_dataframe().set_index('Variable').rename({"Importance": f"label={name}"}, axis=1).loc[feature_names] for name, cimp in conditional_feature_importances.values.items()]
    all_df = pd.concat([imp_df, *cimp_df], axis=1).sort_values(by='Overall', ascending=False)

    # Set bar width and the number of bars
    width = 0.2  # Width of each bar
    num_columns = len(all_df.columns)  # Number of columns to plot

    # Create figure and primary axis
    #fig, ax1 = plt.subplots(figsize=(30, 6), dpi=200)

    # Plot each set of bars with different positions
    x = np.arange(len(all_df))  # Position for each feature

    # Plot conditional importances and Overall importance
    for i, col in enumerate(all_df.columns):  # Exclude the last column (Fluctuation Ratio)
        plt.barh(x + (i * width) - width, all_df[col], width, label=col)

    # Set axis labels and title
    labels = list(all_df.index)
    short_labels = [label.rsplit("_",1)[-1] for label in labels]
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances by Label [{model_name.replace("DecisionTree", "DT")}]')
    #plt.tick_params(axis='x', rotation=60)
    plt.gca().invert_yaxis()
    plt.yticks(x, labels=short_labels)

    plt.legend()
    plt.tight_layout()


def plot_feature_importance_contrast(X, proxy, importances, conditional_feature_importances):
    feature_names = []
    for name, cimp in conditional_feature_importances.values.items():
        feature_names += cimp.top_alpha().feature_names
    feature_names = sorted(set(feature_names))
    
    # Prepare dataframes
    imp_df = importances.as_dataframe().set_index('Variable').rename({'Importance': "Overall"}, axis=1).loc[feature_names]
    cimp_df = [cimp.as_dataframe().set_index('Variable').rename({"Importance": f"label={name}"}, axis=1).loc[feature_names] for name, cimp in conditional_feature_importances.values.items()]
    all_df = pd.concat([imp_df, *cimp_df], axis=1).sort_values(by='Overall', ascending=False)

    # Set bar width and the number of bars
    width = 0.2  # Width of each bar
    num_columns = len(all_df.columns)  # Number of columns to plot

    # Plot each set of bars with different positions
    x = np.arange(len(all_df))  # Position for each feature

    # Plot conditional importances and Overall importance
    for i, col in enumerate(all_df.columns):  # Exclude the last column (Fluctuation Ratio)
        plt.plot(x + (i * width) - width, all_df[col], width, label=col)

    # Set axis labels and title
    labels = list(all_df.index)
    short_labels = [label.rsplit("_",1)[-1] for label in labels]
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances by Label [{model_name.replace("DecisionTree", "DT")}]')
    #plt.tick_params(axis='x', rotation=60)
    #plt.gca().invert_yaxis()
    #plt.yticks(x, labels=short_labels)

    plt.legend()
    plt.tight_layout()

def plot_feature_importance_partial_dependencies_oscillation(model_name, df, top_n=None):#, figsize=None):
    if top_n is not None:
        df = df.iloc[:top_n]

    # Create positions for the bars
    features = df['Variable']
    x_pos = np.arange(len(features))
    
    # Create horizontal lollipop chart
    #plt.hlines(y=range(len(df)), xmin=0, xmax=df['Importance'], color='#5B7BE9', alpha=0.7, linewidth=2)
    #plt.plot(df['Importance'], range(len(df)), "o", markersize=8, color='#5B7BE9', alpha=0.8)
    plt.barh(np.arange(len(df))-0.2, df['Importance'], height=0.4, color='#5B7BE9', alpha=0.8)

    # Add oscillation markers
    feature_names = [f.rsplit("_")[-1] for f in df['Variable'].tolist()]

    # Customize the plot
    plt.yticks(range(len(df)), feature_names)
    plt.xlabel('Value', fontsize=12)

    # Add a second x-axis for oscillation
    plt.gca().invert_yaxis()
    ax1 = plt.gca()
    
    ax2 = ax1.twiny()
    #ax2.plot(df['Oscillation'], range(len(df)), "D", markersize=7, color='#47B39C', alpha=0.8)
    ax2.barh(np.arange(len(df))+0.2, df['Oscillation'], height=0.4, color='#47B39C', alpha=0.8)
    ax2.set_xlim(0, 1)
    #ax2.set_xlim(ax1.get_xlim())

    # Set labels and titles
    ax1.set_xlabel('Importance', color='#5B7BE9', fontsize=12)
    ax2.set_xlabel('PD Oscillation', color='#47B39C', fontsize=12)
    plt.title(f'{model_name}', fontsize=14, pad=20)
    ax1.tick_params(axis='x', colors='#5B7BE9')
    ax2.tick_params(axis='x', colors='#47B39C')
    ax2.grid(False)
    # Add legend
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#5B7BE9', label='Importance', 
            markerfacecolor='#5B7BE9', markersize=8, linewidth=2),
        Line2D([0], [0], color='#47B39C', label='Oscillation',
            markerfacecolor='#47B39C', markersize=7, linewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='lower right')


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_partial_dependence_oscilation(partial_dependencies, top_n=None, subplots=None, figsize=None, feature_names=None):
    if figsize is None:
        figsize = (20, 5)

    ncols = 5
    if feature_names is None:
        feature_names = partial_dependencies.feature_names

    if top_n is None:
        top_n = min(ncols, len(feature_names))

    if subplots is None:
        subplots = ((top_n + ncols - 1)//ncols, ncols)

    _, axs = plt.subplots(1, top_n, figsize=figsize, dpi=100)
    color_map = cm.get_cmap('Blues')

    if top_n == 1:
        axs = [axs]  # Asegura que axs sea iterable incluso con un solo subplot

    for i, feature_name in enumerate(feature_names[:top_n]):
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='individual')

        indice_oscilacion_normalizados = []
        for c in individuals:
            derivada = np.diff(c)
            cambios_signo = np.sum(np.diff(np.sign(derivada)) != 0)
            indice_oscilacion_normalizados.append(cambios_signo / len(c))

        # Calcular el score medio
        score = np.mean(indice_oscilacion_normalizados)

        # Gráfico de histograma
        axs[i].hist(indice_oscilacion_normalizados, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Configuración de los ejes y títulos
        axs[i].grid(True, linestyle='--', alpha=0.7)  # Línea de cuadrícula sutil
        axs[i].set_ylabel("Frequency", fontsize=12, labelpad=10)
        axs[i].set_xlabel("Oscillation Index", fontsize=12, labelpad=10)
        axs[i].set_title(f"Feature: {feature_name} | O={score:.3f}", fontsize=12, pad=15)
        
        # Limitar los ejes x entre 0 y 1
        axs[i].set_xlim([0, 1])
        
        # Asegurar que las etiquetas de los ticks estén bien espaciadas
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))

    # Ajustar el espaciado entre subplots
    plt.tight_layout()


def plot_partial_dependence_with_std(partial_dependencies, top_n=None, figsize=None, feature_names=None):
    if figsize is None:
        figsize = (20, 5)
    
    ncols = 5
    if feature_names is None:
        feature_names = partial_dependencies.feature_names

    if top_n is None:
        top_n = min(ncols, len(feature_names))

    fig, axs = plt.subplots(1, top_n, figsize=figsize, dpi=100)
    color_map = cm.get_cmap('Blues')

    if top_n == 1:
        axs = [axs]  # Asegura que axs sea iterable incluso con un solo subplot

    for i, feature_name in enumerate(feature_names[:top_n]):
        # Obtenemos los valores individuales y la malla de valores (grid_values)
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='individual')
        grid_values = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='grid_values')
        average = partial_dependencies.get_value(feature_name=feature_name, label=0, data_type='average')

        # Convertimos los individuos en una matriz para facilitar el cálculo de la desviación estándar
        individuals_matrix = np.array(individuals)
        
        # Calculamos la desviación estándar en cada punto de grid_values
        std_dev = np.std(individuals_matrix, axis=0)

        # Graficamos la curva promedio
        axs[i].plot(grid_values, average, color='blue', linewidth=2, label='Average')

        # Graficamos la banda de confianza (curva promedio ± desviación estándar)
        axs[i].fill_between(grid_values, average - std_dev, average + std_dev, color='blue', alpha=0.1, label='Std Dev')

        # Ajustamos etiquetas y título
        short_feature_name = feature_name.split('_')[-1]
        axs[i].set_xlabel('Grid Values')
        axs[i].set_ylabel('Partial Dependence')
        axs[i].set_title(f"[feature={short_feature_name}]")
        axs[i].legend()
        
        # Limitamos el número de ticks en el eje X
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    return fig, axs


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from collections import Counter

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
