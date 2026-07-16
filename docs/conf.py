"""Sphinx configuration for the wave-measure documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version

# Make the package importable (src layout) and skip hardware probing on build.
sys.path.insert(0, os.path.abspath("../src"))
os.environ.setdefault("WAVE_MEASURE_SKIP_DETECT", "1")

# -- Project information -----------------------------------------------------

project = "wave-measure"
author = "AL-255"
copyright = "2025-2026, AL-255"

try:
    release = version("wave-measure")
except PackageNotFoundError:  # not installed; fall back to the source version
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# NumPy-style docstrings.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Autodoc.
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Reference bare names (`Waveform`) as Python objects.
default_role = "py:obj"

# Cross-project references.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# MyST (Markdown) extensions.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"wave-measure {release}"
html_static_path = ["_static"]
