"""Sphinx configuration for the separatix documentation."""

from __future__ import annotations

from importlib.metadata import version

project = "separatix"
author = "Niklas Melton"
copyright = "2026, Niklas Melton"
release = version("separatix")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "separatix documentation"
html_logo = "../img/separatix_logo_transparent.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/NiklasMelton/Separatix/",
    "source_branch": "develop",
    "source_directory": "docs/",
}

linkcheck_ignore = [
    r"https://test\.pypi\.org/.*",
]
