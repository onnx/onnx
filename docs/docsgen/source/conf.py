import os
import sys
import warnings

import pydata_sphinx_theme

import onnx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from github_link import make_linkcode_resolve  # noqa


# -- Project information -----------------------------------------------------

project = 'ONNX'
copyright = '2022'
author = 'ONNX'
version = onnx.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx_exec_code',
    'onnx_sphinx',
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'
coverage_show_missing_items = True
onnx_doc_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "onnx_doc_folder")

# Navbar
html_static_path = ["_static"]

# -- Options for HTML output -------------------------------------------------
html_favicon = 'onnx-favicon.png'
html_theme = "pydata_sphinx_theme"
html_logo = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../onnx-horizontal-color.png")
html_theme_options = {
   "logo": {
      "image_dark": "onnx-horizontal-white.png",
   }
}

# -- Options for graphviz ----------------------------------------------------

graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# The name of the Pygments (syntax highlighting) style to use ----------------
pygments_style = 'sphinx'

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': ['examples'],
    'gallery_dirs': ['auto_examples', 'auto_tutorial'],
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
    'binder': {
        'org': 'onnx',
        'repo': '.',
        'notebooks_dir': 'auto_examples',
        'binderhub_url': 'https://mybinder.org',
        'branch': 'master',
        'dependencies': './requirements.txt'
    },
}

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Setup actions -----------------------------------------------------------


blog_root = ""

html_css_files = ['sample.css']

html_sidebars = {}
language = "en"

mathdef_link_only = True

# \\usepackage{eepic}



intersphinx_mapping.update({
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'python': (
        'https://docs.python.org/{.major}'.format(sys.version_info),
        None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
})


nblinks = {
}

warnings.filterwarnings("ignore", category=FutureWarning)
