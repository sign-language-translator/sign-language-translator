# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import sign_language_translator

# -- Project information -----------------------------------------------------

project = "Sign Language Translator"
author = "Mudassar Iqbal"
copyright = f"2023, {author}"

version = sign_language_translator.__version__
release = sign_language_translator.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_rtd_dark_mode",
]

default_dark_mode = False

# root_doc = "introduction"
source_encoding = "utf-8"

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    # now write a cross-reference like :py:func:`io.open`
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

from typing import Optional

import requests


def linkcode_resolve(domain, info) -> Optional[str]:
    """
    Generates a URL for the given domain and module information.

    Parameters:
        domain (str): The domain of the link.
        info (dict): The module information.

    Returns:
        str | None: The generated URL or None if the URL is not valid.
    """

    if domain != "py":
        return None
    if not info["module"]:
        return None

    filename = info["module"].replace(".", "/")
    base_url = "https://github.com/sign-language-translator/sign-language-translator/blob/main/"
    for slug in [filename, f"{filename}/__init__.py", f"{filename}.py"]:
        url = base_url + slug
        try:
            if requests.head(base_url + slug, timeout=20).status_code == 200:
                return url
        except:
            pass

    return None


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    # 'logo_only': True,
    # "logo": {
    #     "text": "Sign Language Translator",
    # },
    # "show_nav_level": 2,
}
# html_logo = "https://avatars.githubusercontent.com/u/130464523"
html_logo = "https://github.com/sign-language-translator/sign-language-translator/assets/118578823/d4723333-3d25-413d-83a1-a4bbdc8da15a"
