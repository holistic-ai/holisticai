def generate_html(data):
    html_template = """
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diagrama DatasetDict</title>
        <style>
            .container {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .node {{
                border: 2px solid #000;
                padding: 10px;
                margin: 10px;
                text-align: left;
                display: inline-block; 
                background-color: #F0F8FF; 
            }}
            .datasets-container {{
                display: flex;
                flex-wrap: wrap;
            }}
            .dataset {{
                border: 2px solid #007bff;
                padding: 10px;
                margin: 10px;
                text-align: left;
                border-radius: 5px;
                display: inline-block;
                white-space: nowrap;
                background-color: #F5FFFA; 
                box-sizing: border-box;
            }}
            .title {{
                font-weight: bold;
                margin-bottom: 10px;
                text-align: left;
                white-space: nowrap;
            }}
            ul {{
                list-style-type: disc;
                padding-left: 20px;
                text-align: left;
                margin: 0;
                white-space: normal;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {content}
        </div>
    </body>
    """

    dataset_template = """
    <div class="dataset" style="width: {width}px;">
        <div class="title">{type}</div>
        <ul>
            <li>features: [ {features} ]</li>
            <li>num_rows: {num_rows}</li>
        </ul>
    </div>
    """

    dataset_in_datasetdict_template = """
    <div class="dataset" style="width: {width}px;">
        <div class="title">{name}: {type}</div>
        <ul>
            <li>features: [{features}]</li>
            <li>num_rows: {num_rows}</li>
        </ul>
    </div>
    """

    if 'DatasetDict' in data:
        node_template = """
        <div class="node">
            <div class="title">DatasetDict</div>
            <div class="datasets-container">
                {datasets}
            </div>
        </div>
        """
        
        max_text_length = max(
            len(f"features: [ {' , '.join(dataset['features'])} ] num_rows: {dataset['num_rows']}") 
            for dataset in data['DatasetDict']
        ) * 8 + 40
        
        datasets_html = ""
        for dataset in data['DatasetDict']:
            datasets_html += dataset_in_datasetdict_template.format(
                name=dataset['name'],
                type=dataset['type'],
                features=" , ".join(dataset['features']),
                num_rows=dataset['num_rows'],
                width=max_text_length
            )
        
        content_html = node_template.format(datasets=datasets_html)
    elif 'Dataset' in data:
        max_text_length = (
            len(f"features: [ {' , '.join(data['Dataset']['features'])} ] num_rows: {data['Dataset']['num_rows']}")
        ) * 8 + 40
        content_html = dataset_template.format(
            type='Dataset',
            features=" , ".join(data['Dataset']['features']),
            num_rows=data['Dataset']['num_rows'],
            width=max_text_length
        )

    final_html = html_template.format(content=content_html)
    return final_html

if __name__ == "__main__":
    # Ejemplo de uso para DatasetDict
    data_dict = {
        'DatasetDict': [
            {'type': 'Dataset', 'name': 'train', 'features': ['x', 'y', 'group_a', 'group_b'], 'num_rows': 2480},
            {'type': 'Dataset', 'name': 'valid', 'features': ['x', 'y', 'p_attr'], 'num_rows': 1620},
            {'type': 'Dataset', 'name': 'test', 'features': ['x', 'y', 'group_a', 'group_b'], 'num_rows': 1620}
        ]
    }

    html_output_dict = generate_html(data_dict)
    with open('datasetdict.html', 'w') as file:
        file.write(html_output_dict)

    # Ejemplo de uso para Dataset
    data_single = {
        'Dataset': {
            'name': 'train', 
            'features': ['x', 'y', 'group_a', 'group_b'], 
            'num_rows': 2480
        }
    }

    html_output_single = generate_html(data_single)
    with open('dataset.html', 'w') as file:
        file.write(html_output_single)
