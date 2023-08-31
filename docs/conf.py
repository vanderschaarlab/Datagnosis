# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# stdlib
import subprocess

from datagnosis.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "datagnosis"
copyright = "2023, vanderschaar-lab"
author = "vanderschaar-lab"
release = __version__

subprocess.run(
    [
        "sphinx-apidoc",
        "--ext-autodoc",
        "--ext-doctest",
        "--ext-mathjax",
        "--ext-viewcode",
        "-e",  # put documentation for each module on its own page
        "-T",  # don't create a table of contents file
        "-M",  # put module documentation before submodule documentation
        "-F",  # generate a full project with sphinx-quickstart
        "-P",  # include "_private" modules
        "-f",  # overwrite existing files
        "-o",  # directory to place all output - kwarg
        "generated",  # directory to place all output
        "-t",  # template directory for template files -kwarg
        "_templates",  # template directory for template files
        "../src/datagnosis/",  # path to module to document
    ]
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "m2r2",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "nbsphinx",
    "sphinxemoji.sphinxemoji",
]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "inherit_docstrings": True,
    "private-members": False,
}

add_module_names = False
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

nbsphinx_execute = "never"

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
"""

autodoc_mock_imports = [
    "cloudpickle",
    "lifelines",
    "loguru",
    "pgmpy",
    "pycox",
    "pykeops",
    "pyod",
    "scikit-learn",
    "sklearn",
    "pytorch_lightning",
    "scipy",
]

autodoc_mock_imports = autodoc_mock_imports
