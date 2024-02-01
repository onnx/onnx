# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0


# type: ignore
import os
import sys
import warnings

import onnx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# -- Project information -----------------------------------------------------

author = "ONNX"
copyright = "2024"
project = "ONNX"
release = onnx.__version__
version = onnx.__version__

# define the latest opset to document,
# this is meant to avoid documenting opset not released yet
max_opset = onnx.helper.VERSION_TABLE[-1][2]

# define the latest opset to document for every opset
_opsets = [t for t in onnx.helper.VERSION_TABLE if t[2] == max_opset][-1]
max_opsets = {
    "": max_opset,
    "ai.onnx.ml": _opsets[3],
    "ai.onnx.training": _opsets[4],
}

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "onnx_sphinx",
    "sphinx_copybutton",
    "sphinx_exec_code",
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

coverage_show_missing_items = True
exclude_patterns = []
graphviz_output_format = "svg"
html_css_files = ["css/custom.css"]
html_favicon = "onnx-favicon.png"
html_sidebars = {}
html_static_path = ["_static"]
html_theme = "furo"
language = "en"
mathdef_link_only = True
master_doc = "index"
onnx_doc_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "operators")
pygments_style = "default"
source_suffix = [".rst", ".md"]
templates_path = ["_templates"]

html_context = {
    "default_mode": "auto",  # auto: the documentation theme will follow the system default that you have set (light or dark)
}

html_theme_options = {
    "light_logo": "onnx-horizontal-color.png",
    "dark_logo": "onnx-horizontal-white.png",
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
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
