import os
import sys
import warnings

sys.path.append(os.path.abspath('exts'))
# from github_link import make_linkcode_resolve  # noqa


# -- Project information -----------------------------------------------------

project = 'ONNX'
copyright = '2022'
author = 'Yesha Thakkar'
version = "0.1"
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
    'sphinxcontrib.blockdiag',
    'mlprodict.npy.xop_sphinx',
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'
coverage_show_missing_items = True

# -- Options for HTML output -------------------------------------------------

html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_logo = "../../images/ONNX_ICON.png"

# Navbar
html_theme_options = {
    'navbar_title': "ONNX Docs",
}
html_static_path = ["_static"]

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


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app



blog_root = ""

html_css_files = ['sample.css']

html_sidebars = {}
language = "en"

mathdef_link_only = True

# \\usepackage{eepic}

#Code to generate include.rst
files = os.listdir('../onnx_doc_folder')

with open('include.rst', 'w') as file:
    for f in files:
        if (f != 'index.rst'):
            file.write('.. include:: ../onnx_doc_folder/' + f + '\n')



intersphinx_mapping.update({
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mlprodict':
        ('http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
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
