from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import antsnormflows  # noqa: E402


project = "ANTsNormalizingFlows"
author = "ANTsX"
copyright = "2026, ANTsX"
release = antsnormflows.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_context = {
    "display_github": True,
    "github_user": "ANTsX",
    "github_repo": "ANTsNormalizingFlows",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
