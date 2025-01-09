# ruff: noqa: E501

"""Sphinx configuration file."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

from autoplex import __version__

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "autoplex"
copyright = "2024, autoplex development team"  # noqa: A001
#author = "autoplex development team"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store", "test*.py", "_build"]

# mermaid settings
mermaid_version = "11.2.0"
mermaid_output_format = 'raw'
mermaid_params = ['--theme', 'base']

myst_heading_anchors = 4  # enable headings as link targets
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]

# use type hints
autodoc_typehints = "description"
# autoclass_content = "both"
# autodoc_member_order = "bysource"

# better napoleon support
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

mathjax3_config = {
    "tex": {
        "macros": {
            "N": "\\mathbb{N}",
            "floor": ["\\lfloor#1\\rfloor", 1],
            "bmat": ["\\left[\\begin{array}"],
            "emat": ["\\end{array}\\right]"],
        }
    }
}
latex_elements = {
    "preamble": r"""\newcommand\N{\mathbb{N}}
\newcommand\floor[1]{\lfloor#1\rfloor}
\newcommand{\bmat}{\left[\begin{array}}
\newcommand{\emat}{\end{array}\right]}
"""
}
language = "en"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
suppress_warnings = ['toc.excluded', 'toc.not_readable']

# autodoc options
autosummary_imported_members = False
autodoc_preserve_defaults = True
autoclass_content = "class"
autodoc_member_order = "bysource"

python_use_unqualified_type_names = True

# don't overwrite summary but generate them if they don't exist
autosummary_generate = True
autosummary_generate_overwrite = True

# numpydoc options
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False
numpydoc_xref_param_type = True

# sphinx-panels shouldn't add bootstrap css as the pydata-sphinx-theme already loads it
panels_add_bootstrap_css = False

# control pydantic model docs
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_show_config = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/autoplex_logo.png"
html_favicon = "_static/autoplex_favicon.png"
html_theme_options = {
    "repository_provider": "github",
    "repository_url": "https://github.com/JaGeo/autoplex",
    "use_repository_button": True,
    "use_issues_button": True,
}

# html_static_path = ["_static"]

# hide sphinx footer
html_show_sphinx = False
html_show_sourcelink = False

# downgrad mathjax
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
mathjax2_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'processEscapes': True,
        'ignoreClass': 'document',
        'processClass': 'math|output_area',
    }
}
html_title = "autoplex"
