import os
import sys

import uscrn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from fancy_version import VersionInfo

project = "uscrn"
copyright = "2023\u20132025"
author = "zmoon"

version = uscrn.__version__.split("+")[0]
release = uscrn.__version__
footer_version = VersionInfo(uscrn).fancy_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.doctest",
    # "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    # "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    # "sphinx.ext.viewcode",
    # "sphinx_jinja",
    "myst_nb",
    "sphinx_click",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    # "pint": ("https://pint.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

exclude_patterns = ["_build"]

html_title = "USCRN Python"
html_theme = "sphinx_book_theme"

html_theme_options = {
    "path_to_docs": "docs/",
    "repository_url": "https://github.com/zmoon/uscrn",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "extra_footer": f"""\
    <span style="font-size: 0.8em;">uscrn version in this docs build:
    <strong>{footer_version}</strong>.</span>
    """,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_typehints = "description"
autosummary_generate = True

nb_execution_mode = "cache"
nb_execution_excludepatterns = ["examples/daily.ipynb"]
nb_execution_raise_on_error = True
nb_execution_timeout = 60
