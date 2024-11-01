import matplotlib.pyplot as plt
import numpy as np


def plot_radar_metrics(df, model_names, figsize=(16, 8)):
    color_map = {"efficacy": "blue", "global": "green", "local": "red", "surrogate": "purple"}

    # Extraer las métricas y sus tipos
    def replace_metric_name(metric_name):
        if metric_name == "Feature Rank Stability":
            return "Position Consistency"
        if metric_name == "Feature Importance Stability":
            return "Importance Stability"
        return metric_name

    df.index = [replace_metric_name(metric) for metric in df.index]
    metrics = df.index
    metric_types = df["metric_type"]

    # Crear una lista modificada de nombres de métricas basada en si Reference es 0
    modified_metrics = []
    for i, metric in enumerate(metrics):
        if df["Reference"][i] == 0:
            modified_metrics.append(f"1-{metric}")  # Cambiar el nombre de la métrica
        else:
            modified_metrics.append(metric)

    # Crear el gráfico de radar
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    # Número de variables (métricas)
    N = len(metrics)

    # Ángulos para cada métrica
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Crear barras radiales para colorear secciones circulares
    width = 2 * np.pi / N  # Ancho de cada barra
    for i in range(N):
        color = color_map[metric_types[i]]  # Color basado en el tipo de métrica
        ax.bar(angles[i], 1, width=width, color=color, alpha=0.2, edgecolor="none")  # Dibujar secciones circulares

    # Función para dibujar los modelos sin colorear las regiones
    def plot_model(ax, model_data, model_name):
        # Ajustar los valores según la columna de referencia (1 - métrica si Reference es 0)
        adjusted_values = []
        for i in range(N):
            if df["Reference"][i] == 0:
                adjusted_values.append(1 - model_data[i])  # Invertir valor si Reference es 0
            else:
                adjusted_values.append(model_data[i])
        adjusted_values += adjusted_values[:1]  # Completar el círculo

        # Dibujar el radar chart con un borde general
        ax.plot([*angles, angles[0]], adjusted_values, linewidth=2, linestyle="solid", label=model_name)

    # Dibujar todos los modelos en un solo gráfico
    for model in model_names:
        plot_model(ax, df[model], model)

    # Añadir las etiquetas de las métricas en los ángulos correctos usando los nombres modificados
    ax.set_xticks(angles)
    ax.set_xticklabels(modified_metrics, fontsize=15)

    # Ajustar el rango del gráfico
    ax.set_ylim(0, 1)

    # Añadir leyenda en la parte superior izquierda
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 1.1))

    # Ajustar el layout y mostrar gráfico
    plt.tight_layout()
