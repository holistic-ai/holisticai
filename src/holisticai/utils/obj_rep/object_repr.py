import os
from abc import abstractmethod


class ReprObj:
    _theme = "blue"

    @abstractmethod
    def repr_info(self):
        return {}

    @property
    def _repr_html_(self):
        def generate_html():
            return generate_html_for_generic_object(self.repr_info(), feature_columns=5, theme="blue")

        return generate_html

    def repr_html(self):
        return generate_html_for_generic_object(self.repr_info(), feature_columns=5, theme=self._theme)

    def _repr_mimebundle_(self, **_):
        return {"text/html": self.repr_html(), "text/plain": self.repr_info()}


class DatasetReprObj(ReprObj):
    _theme = "blue"


class BMReprObj(ReprObj):
    _theme = "orange"


class PipelineReprObj(ReprObj):
    _theme = "green"


def generate_html_for_generic_object(obj, feature_columns=5, theme="blue"):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_repr.css")

    with open(css_path) as f:
        css_template = f.read()

    html_template = """
    <style>
        {css_template}
    </style>
    <div style="display: flex;">
    <div class="generic-object-container {theme} first-level">
        <div class="generic-object-content">
            <div class="generic-object-header {theme}" onclick="toggleCollapse(this)">
                <button class="toggle-button {theme}">[-]</button> {header}
            </div>
            <div class="generic-object-body">
                {attributes}
                {nested_objects}
            </div>
        </div>
    </div>
</div>
    <script>
    function toggleCollapse(element) {{
        var body = element.nextElementSibling;
        var button = element.querySelector(".toggle-button");
        if (body.classList.contains('hidden')) {{
            body.classList.remove('hidden');
            button.textContent = "[-]";
        }} else {{
            body.classList.add('hidden');
            button.textContent = "[+]";
        }}
    }}
    </script>
    """

    name = obj.get("name", "N/A")
    obj_type = obj.get("dtype", "N/A")
    attributes = obj.get("attributes", {})
    metadata = obj.get("metadata", None)
    subtitle = obj.get("subtitle", None)
    nested_objects = obj.get("nested_objects", [])

    attributes_html = ""
    if subtitle is not None:
        attributes_html += f'<div class="attribute-list {theme}"><center>{subtitle}</center></div>'

    if attributes_html != "" and (attributes != {} or metadata is not None):
        attributes_html += "<hr>"

    for key, value in attributes.items():
        value_ = value
        if isinstance(value, list):
            value_ = ", ".join(map(str, value))
        attributes_html += f'<div class="attribute-list {theme}"><strong>{key.capitalize()}</strong>: {value_}</div>'

    if isinstance(metadata, str):
        attributes_html += f'<div class="attribute-list {theme}"><strong>Metadata</strong>: {metadata}</div>'
    elif isinstance(metadata, dict):
        for key, value in metadata.items():
            value_ = value
            if isinstance(value, list):
                value_ = ", ".join(map(str, value))
            attributes_html += (
                f'<div class="attribute-list {theme}"><strong>{key.capitalize()}</strong>: {value_}</div>'
            )

    nested_objects_html = ""
    for nested_obj in nested_objects:
        nested_objects_html += generate_html_for_generic_object(nested_obj, feature_columns, theme)

    header = f"[{obj_type}]" if name in ("N/A", "") else f"{name} [{obj_type}]"
    return html_template.format(
        header=header,
        attributes=attributes_html,
        nested_objects=nested_objects_html,
        css_template=css_template,
        theme=theme,
    )
