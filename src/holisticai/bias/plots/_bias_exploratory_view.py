import ipywidgets as widgets  # type: ignore
import matplotlib.pyplot as plt
from holisticai.bias.plots._bias_exploratory_plots import group_pie_plot, histogram_plot
from holisticai.bias.plots._bias_multiclass_plots import accuracy_bar_plot, frequency_matrix_plot, frequency_plot
from IPython.display import display


def binary_classification_data_exploration(dataset):
    from io import StringIO

    from IPython.core.display import HTML
    from ipywidgets import embed  # type: ignore

    output = widgets.Output()

    def update_plot(change):
        with output:
            output.clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            group_pie_plot(dataset["p_attr"][change["new"]], ax=axs[0])
            axs[0].set_title(f"Protected Attribute/Group: {change['new']}")
            axs[0].grid(True)

            frequency_plot(dataset["p_attr"][change["new"]], dataset["y"], ax=axs[1])
            axs[1].set_title(f"Positive Rate by Group: {change['new']}")
            axs[1].grid(True)

            plt.show()

    protected_attributes = dataset["p_attr"].columns
    dropdown = widgets.Dropdown(options=protected_attributes, value=protected_attributes[0], description="p_attr:")
    dropdown.observe(update_plot, names="value")
    content = widgets.VBox([dropdown, output])
    display(content)
    update_plot({"new": dropdown.value})

    html_stream = StringIO()
    embed.embed_minimal_html(html_stream, views=[content], title="Mi widget")

    # Obtener el contenido HTML como una cadena
    html_content = html_stream.getvalue()

    # Mostrar el HTML
    display(HTML(html_content))


def clustering_data_exploration(dataset):
    output = widgets.Output()

    def update_plot(change):
        with output:
            output.clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            group_pie_plot(dataset["p_attr"][change["new"]], ax=axs[0])
            axs[0].set_title(f"Protected Attribute/Group: {change['new']}")
            axs[0].grid(True)

            histogram_plot(dataset["p_attr"][change["new"]], dataset["y"], ax=axs[1])
            axs[1].set_title(f"Clusters by Group: {change['new']}")
            axs[1].grid(True)

            plt.show()

    protected_attributes = dataset["p_attr"].columns
    dropdown = widgets.Dropdown(options=protected_attributes, value=protected_attributes[0], description="p_attr:")
    dropdown.observe(update_plot, names="value")
    content = widgets.VBox([dropdown, output])
    display(content)
    update_plot({"new": dropdown.value})


def binary_classification_model_exploration(dataset, y_pred):
    output = widgets.Output()

    def update_plot(change):
        with output:
            output.clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # group_pie_plot(dataset['p_attr'][change['new']], ax=axs[0])
            accuracy_bar_plot(dataset["p_attr"][change["new"]], y_pred, dataset["y"], ax=axs[0])
            axs[0].set_title(f"Accuracy Bar Plot: {change['new']}")
            axs[0].grid(True)

            frequency_plot(dataset["p_attr"][change["new"]], y_pred, ax=axs[1])
            axs[1].set_title(f"Positive Rate by Group: {change['new']}")
            axs[1].grid(True)

            plt.show()

    protected_attributes = dataset["p_attr"].columns
    dropdown = widgets.Dropdown(options=protected_attributes, value=protected_attributes[0], description="p_attr:")
    dropdown.observe(update_plot, names="value")
    content = widgets.VBox([dropdown, output])
    display(content)
    update_plot({"new": dropdown.value})


def clustering_model_exploration(dataset, y_pred):
    output = widgets.Output()

    def update_plot(change):
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            frequency_matrix_plot(dataset["p_attr"][change["new"]], y_pred, normalize="group", ax=ax)
            ax.set_title(f"Group-Clusters Frequency: {change['new']}")
            ax.grid(True)

            plt.show()

    protected_attributes = dataset["p_attr"].columns
    dropdown = widgets.Dropdown(options=protected_attributes, value=protected_attributes[0], description="p_attr:")
    dropdown.observe(update_plot, names="value")
    content = widgets.VBox([dropdown, output])
    display(content)
    update_plot({"new": dropdown.value})


def bias_data_exploration(learning_task, dataset):
    if learning_task == "binary_classification":
        return binary_classification_data_exploration(dataset)
    if learning_task == "clustering":
        return clustering_data_exploration(dataset)
    return None


def bias_model_exploration(learning_task, dataset, y_pred):
    if learning_task == "binary_classification":
        return binary_classification_model_exploration(dataset, y_pred)
    if learning_task == "clustering":
        return clustering_model_exploration(dataset, y_pred)
    return None
