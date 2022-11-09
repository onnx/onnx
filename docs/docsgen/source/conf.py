# pylint: disable=W0622
# type: ignore
import os
import sys
import warnings

import onnx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# -- Project information -----------------------------------------------------

author = "ONNX"
copyright = "2022"
project = "ONNX"
release = onnx.__version__
version = onnx.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx_exec_code",
    "sphinx_tabs.tabs",
    "onnx_sphinx",
]

coverage_show_missing_items = True
exclude_patterns = []
graphviz_output_format = "svg"
html_css_files = ["css/custom.css"]
html_favicon = "onnx-favicon.png"
html_logo = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "../../onnx-horizontal-color.png"
)
html_sidebars = {}
html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
language = "en"
mathdef_link_only = True
master_doc = "index"
onnx_doc_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "operators")
pygments_style = "sphinx"
source_suffix = [".rst"]
templates_path = ["_templates"]

html_context = {
    "default_mode": "auto",  # auto: the documentation theme will follow the system default that you have set (light or dark)
}

html_theme_options = {
    "collapse_navigation": True,
    "external_links": [
        {"name": "ONNX", "url": "https://onnx.ai/"},
        {"name": "github", "url": "https://github.com/onnx/onnx"},
    ],
    "github_url": "https://github.com/onnx/onnx",
    "logo": {"image_dark": "onnx-horizontal-white.png"},
    "navbar_center": [],
    "navigation_depth": 5,
    "page_sidebar_items": [],  # default setting is: ["page-toc", "edit-this-page", "sourcelink"],
    "show_nav_level": 0,
    "show_prev_next": True,
    "show_toc_level": 0,
}

intersphinx_mapping = {
    "https://docs.python.org/": None,
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples", "auto_tutorial"],
    "capture_repr": ("_repr_html_", "__repr__"),
    "ignore_repr_types": r"matplotlib.text|matplotlib.axes",
    "binder": {
        "org": "onnx",
        "repo": ".",
        "notebooks_dir": "auto_examples",
        "binderhub_url": "https://mybinder.org",
        "branch": "master",
        "dependencies": "./requirements.txt",
    },
}

warnings.filterwarnings("ignore", category=FutureWarning)
