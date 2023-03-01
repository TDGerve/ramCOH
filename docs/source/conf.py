# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ramCOH"
copyright = "2023, Thomas van Gerve"
author = "Thomas van Gerve"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "csaps": ("https://csaps.readthedocs.io/en/v1.1.0", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None)
}
intersphinx_disabled_reftypes = ["*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["../_static"]
