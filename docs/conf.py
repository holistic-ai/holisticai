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


# -- Project information -----------------------------------------------------

project = "holisticai"
copyright = "2022, Holistic AI"
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
    "nbsphinx",
    "sphinx_copybutton",
]

# autodoc options
autodoc_default_options = {"members": True, "inherited-members": True}

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

html_logo = "holistic_ai.png"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# Theme options
# html_theme_options = {'logo_only': True,'display_version': True}


import os
import shutil
import sys

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
napoleon_custom_sections = ["Interpretation", "Description", "Parameters", "Returns"]
