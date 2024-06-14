def generate_css():
    return """
    .container {
        font-family: Arial, sans-serif;
        margin: 20px;
    }
    .node {
        border: 2px solid #000;
        padding: 10px;
        margin: 10px;
        text-align: left;
        display: inline-block;
        background-color: #E0FFFF;
    }
    .datasets-container {
        display: flex;
        flex-wrap: wrap;
    }
    .dataset {
        border: 2px solid #007bff;
        padding: 10px;
        margin: 10px;
        text-align: left;
        border-radius: 5px;
        display: inline-block;
        white-space: nowrap;
        background-color: #E0FFFF;
        box-sizing: border-box;
    }
    .title {
        font-weight: bold;
        color: #222; /* Letras más negras */
        margin-bottom: 10px;
        text-align: left;
        white-space: nowrap;
    }
    ul {
        list-style-type: disc;
        padding-left: 20px;
        text-align: left;
        margin: 0;
        white-space: normal;
        color: #222; /* Letras más negras */
    }
    .groupbydataset {
        border: 2px solid #007bff; /* Dark red */
        padding: 10px;
        margin: 10px;
        text-align: left;
        border-radius: 5px;
        display: inline-block;
        white-space: nowrap;
        background-color: #E0FFFF; /* Light pink */
        box-sizing: border-box;
    }
    """


def generate_html(data):
    def generate_datasetdict_template(datasets_html):
        return f"""
        <div class="node">
            <div class="title">DatasetDict</div>
            <div class="datasets-container">
                {datasets_html}
            </div>
        </div>
        """

    def generate_dataset_template(features, num_rows, width):
        return f"""
        <div class="dataset" style="width: {width}px;">
            <div class="title">Dataset</div>
            <ul>
                <li>features: [ {features} ]</li>
                <li>num_rows: {num_rows}</li>
            </ul>
        </div>
        """

    def generate_groupbydataset_template(grouped_names, features, ngroups, width):
        return f"""
        <div class="groupbydataset" style="width: {width}px;">
            <div class="title">GroupByDataset</div>
            <ul>
                <li>group: [ {grouped_names} ]</li>
                <li>features: [ {features} ]</li>
                <li>num_groups: {ngroups}</li>
            </ul>
        </div>
        """

    def generate_dataset_in_datasetdict_template(name, features, num_rows, width):
        return f"""
        <div class="dataset" style="width: {width}px;">
            <div class="title">{name}</div>
            <ul>
                <li>features: [{features}]</li>
                <li>num_rows: {num_rows}</li>
            </ul>
        </div>
        """

    def generate_html_template(content):
        return f"""
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Diagrama DatasetDict</title>
            <style>
                {generate_css()}
            </style>
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        """

    if "DatasetDict" in data:
        max_text_length = (
            max(
                len(f"features: [ {' , '.join(dataset['features'])} ] num_rows: {dataset['num_rows']}")
                for dataset in data["DatasetDict"]
            )
            * 8
            + 40
        )

        datasets_html = ""
        for dataset in data["DatasetDict"]:
            datasets_html += generate_dataset_in_datasetdict_template(
                name=dataset["name"],
                features=" , ".join(dataset["features"]),
                num_rows=dataset["num_rows"],
                width=max_text_length,
            )

        content_html = generate_datasetdict_template(datasets_html)

    elif "Dataset" in data:
        max_text_length = (
            len(f"features: [ {' , '.join(data['Dataset']['features'])} ] num_rows: {data['Dataset']['num_rows']}")
        ) * 8 + 40
        content_html = generate_dataset_template(
            features=" , ".join(data["Dataset"]["features"]),
            num_rows=data["Dataset"]["num_rows"],
            width=max_text_length,
        )

    elif "GroupByDataset" in data:
        max_text_length = (
            len(
                f"group: [ {' , '.join(data['GroupByDataset']['grouped_names'])} ] features: [ {' , '.join(data['GroupByDataset']['features'])} ] ngroups: {data['GroupByDataset']['ngroups']}"
            )
        ) * 8 + 40
        content_html = generate_groupbydataset_template(
            grouped_names=" , ".join(data["GroupByDataset"]["grouped_names"]),
            features=" , ".join(data["GroupByDataset"]["features"]),
            ngroups=data["GroupByDataset"]["ngroups"],
            width=max_text_length,
        )

    return generate_html_template(content_html)


if __name__ == "__main__":
    # Ejemplo de uso para DatasetDict
    data_dict = {
        "DatasetDict": [
            {"type": "Dataset", "name": "train", "features": ["x", "y", "group_a", "group_b"], "num_rows": 2480},
            {"type": "Dataset", "name": "valid", "features": ["x", "y", "p_attr"], "num_rows": 1620},
            {"type": "Dataset", "name": "test", "features": ["x", "y", "group_a", "group_b"], "num_rows": 1620},
        ]
    }

    html_output_dict = generate_html(data_dict)
    with open("datasetdict.html", "w") as file:
        file.write(html_output_dict)

    # Ejemplo de uso para Dataset
    data_single = {"Dataset": {"name": "train", "features": ["x", "y", "group_a", "group_b"], "num_rows": 2480}}

    html_output_single = generate_html(data_single)
    with open("dataset.html", "w") as file:
        file.write(html_output_single)
