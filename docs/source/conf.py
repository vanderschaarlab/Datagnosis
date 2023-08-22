# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# stdlib
import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "datagnosis"
copyright = "2023, Rob Davis"
author = "Rob Davis"
release = "0.0.1"


subprocess.run(
    [
        "sphinx-apidoc",
        "--ext-autodoc",
        "--ext-doctest",
        "--ext-mathjax",
        "--ext-viewcode",
        "-e",
        "-T",
        "-M",
        "-F",
        "-P",
        "-f",
        "-o",
        "generated",
        "-t",
        "_templates",
        "../src/datagnosis/",
    ]
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "nbsphinx",
    "sphinxemoji.sphinxemoji",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_member_order = "bysource"
