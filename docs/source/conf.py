# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# z
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sys
import os
import shutil
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
sys.path.insert(0, os.path.abspath('.'))
import utils.xai_image_plots as xai_utils
import inspect

os.makedirs('_static/images', exist_ok=True)

for name, obj in inspect.getmembers(xai_utils):
    if inspect.isfunction(obj) and name.startswith('image_'):
        obj()

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            with open(notebook_path, 'wt') as f:
                nbformat.write(nb, f)
            print(f"Executed: {notebook_path}")
            print(f"Output: {notebook_path}")
        except Exception as e:
            print(f"Error executing the notebook {notebook_path}: {e}")

def run_all_notebooks(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                run_notebook(notebook_path)

sys.path.insert(0, os.path.abspath(".."))
work_dir = '/'.join(os.getcwd().split("/")[:-2])
src_path = os.path.join(work_dir,'src')
sys.path.insert(0, src_path)

bias_tutorial_path = os.path.join(work_dir, 'tutorials/bias')
dataset_tutorial_path = os.path.join(work_dir, 'tutorials/datasets')
xai_tutorial_path = os.path.join(work_dir, 'tutorials/explainability')

def copy_folder(origen, destino):
    try:
        if not os.path.exists(destino):
            os.makedirs(destino)
        shutil.copytree(origen, destino, dirs_exist_ok=True)
        #run_all_notebooks(destino)
        print(f"Folder copied from {origen} to {destino} sucessfully.")
    except Exception as e:
        print(f"Error when trying to copy folder: {e}")

for path in [bias_tutorial_path, dataset_tutorial_path, xai_tutorial_path]:
    dirname = os.path.basename(path)
    copy_folder(path, os.path.join(os.getcwd(), 'gallery', 'tutorials', dirname))

# -- Project information -----------------------------------------------------

project = "holisticai"
copyright = "2024, Holistic AI"
author = "Holistic AI"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    'sphinx.ext.viewcode',
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.youtube",
    'sphinx.ext.mathjax',
]

nbsphinx_allow_errors = True  # Permitir errores en los notebooks
nbsphinx_execute = 'never'  # Puede ser 'auto', 'always', o 'never'

html_show_sourcelink = False
# autodoc options
autodoc_default_options = {"members": True, "inherited-members": False}

# Turn on autosummary
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "generated/*",
    ".ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
numfig=True

html_logo = "hai_logo.svg"
html_favicon = "holistic_ai.png"

html_theme = "pydata_sphinx_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo": {"image_dark": "https://assets-global.website-files.com/6305e5d42c283515c3e71b8c/63d771efd50a073bd66193f0_Holistic-AI-Logo-Horizontal-Dark.svg"},
    "github_url": "https://github.com/holistic-ai/holisticai",
    "twitter_url": "https://twitter.com/holistic_ai",
    "show_version_warning_banner": True,
    "secondary_sidebar_items": [],
#    "announcement": "Visit our website and <a href='https://www.holisticai.com/demo'>schedule a demo</a> with our experts to find out how Holistic AI can help you shield against AI risks.",
    "icon_links": [
        {
            "name": "Community",
            "url": "https://join.slack.com/t/holisticaicommunity/shared_invite/zt-2jamouyrn-BrMfeoBZIHT8HbLzB3P9QQ",  # required
            "icon": "fa-brands fa-slack",
            "type": "fontawesome",
        }
   ],
}

import os
import shutil
import sys

#sys.path.insert(0, os.path.abspath('../src'))

sys.path.insert(0, os.path.abspath(".."))

# make copy of notebooks in docs folder, as they must be here for sphinx to
# pick them up properly.

# Uncomment when tutorials folder is present
"""
NOTEBOOKS_DIR = os.path.abspath("tutorials")
if os.path.exists(NOTEBOOKS_DIR):
    import warnings

    warnings.warn("tutorials directory exists, replacing...")
    shutil.rmtree(NOTEBOOKS_DIR)
shutil.copytree(
    os.path.abspath("../tutorials"),
    NOTEBOOKS_DIR,
)
if os.path.exists(NOTEBOOKS_DIR + "/local_scratch"):
    shutil.rmtree(NOTEBOOKS_DIR + "/local_scratch")
"""

# Custom css
html_css_files = [
    "css/custom_style.css",
]

# Custom section headers
napoleon_custom_sections = ["Interpretation", "Description", "Parameters", "Methods", "Returns"]