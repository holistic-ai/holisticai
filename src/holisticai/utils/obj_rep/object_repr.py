import os


def generate_html_for_generic_object(obj, feature_columns=5):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_repr.css")

    with open(css_path) as f:
        css_template = f.read()

    html_template = """
    <style>
        {css_template}
    </style>
    <div class="generic-object-container">
        <div class="generic-object-content">
            <div class="generic-object-header">{header}</div>
            <div class="generic-object-body">
                {attributes}
                {nested_objects}
            </div>
        </div>
    </div>
    """

    # Extract generic object information
    name = obj.get("name", "N/A")
    obj_type = obj.get("dtype", "N/A").upper()
    attributes = obj.get("attributes", {})
    metadata = obj.get("metadata", None)
    nested_objects = obj.get("nested_objects", [])

    # Generate HTML for attributes
    attributes_html = ""
    for key, value in attributes.items():
        if isinstance(value, list):
            value = ", ".join(map(str, value))  # noqa: PLW2901
        attributes_html += f'<div class="attribute-list">- {key}: {value}</div>'

    if isinstance(metadata, str):
        attributes_html += f'<div class="attribute-list">- Metadata: {metadata}</div>'
    elif isinstance(metadata, dict):
        for key, value in attributes.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))  # noqa: PLW2901
            attributes_html += f'<div class="attribute-list">- {key}: {value}</div>'

    # Generate HTML for nested objects
    nested_objects_html = ""
    for nested_obj in nested_objects:
        nested_objects_html += generate_html_for_generic_object(nested_obj, feature_columns)

    # Fill the main HTML template with attributes and nested objects
    header = f"{obj_type}" if name in ("N/A", "") else f"{name} : {obj_type}"
    html_output = html_template.format(
        header=header, attributes=attributes_html, nested_objects=nested_objects_html, css_template=css_template
    )

    return html_output


if __name__ == "__main__":
    # Example usage
    generic_object = {
        "dtype": "DatasetDict",
        "attributes": {},
        "nested_objects": [
            {
                "dtype": "Dataset",
                "name": "train",
                "attributes": {"Number of Rows": 2480, "Features": ["x", "y", "group_a", "group_b"]},
                "nested_objects": [],
            },
            {
                "dtype": "Dataset",
                "name": "test",
                "attributes": {"Number of Rows": 2480, "Features": ["x", "y", "group_a", "group_b"]},
                "nested_objects": [],
            },
        ],
    }

    # Generate HTML representation for the generic object
    html_output_generic_object = generate_html_for_generic_object(generic_object, feature_columns=5)
