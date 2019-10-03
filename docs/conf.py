# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./neptune-client/'))

# -- Project information -----------------------------------------------------

project = 'Neptune docs'
copyright = '2019, neptune-ml team'
author = 'neptune-ml team'

# The full version, including alpha/beta/rc tags
release = '2.0.0'

# The default language to highlight source code in
highlight_language = 'python3'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    'neptune-client/CODE_OF_CONDUCT.md', 'neptune-client/README.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = '_static/images/others/logo-horizontal.png'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': 'https://docs.neptune.ml/',
    'collapse_navigation': False,
    'style_external_links': False,
    'navigation_depth': 3,
    'prev_next_buttons_location': 'bottom',
    'sticky_navigation': False,
    'titles_only': False,
    'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css'
]

html_favicon = '_static/images/others/favicon.ico'
