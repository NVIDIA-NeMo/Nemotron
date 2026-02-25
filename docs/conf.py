# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

project = "Nemotron"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"

# Version is set by CI via DOCS_VERSION env var (dev or stable)
# Defaults to "dev" for local builds
release = os.environ.get("DOCS_VERSION", "nightly")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks
    "sphinx_design",  # For grid cards and other design elements
    "sphinxcontrib.mermaid",  # For mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
    "attrs_block",  # Enables setting attributes on block elements using {#id .class key=val}
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# Configure MyST to handle mermaid code blocks
myst_fence_as_directive = ["mermaid"]

# -- Options for Mermaid -----------------------------------------------------
# Configure mermaid diagrams
mermaid_version = "latest"  # Use the latest version of mermaid

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Custom static files for Termynal terminal animations -------------------
html_static_path = ["_static"]
html_css_files = [
    "css/termynal.css",
]
html_js_files = [
    "js/termynal.js",
    "js/termynal-init.js",
]

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-NeMo/Nemotron",
            "icon": "fa-brands fa-github",
        }
    ],
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}
html_extra_path = ["project.json", "versions1.json"]

# Github links are now getting rate limited from the Github Actions
linkcheck_ignore = [
    ".*github\\.com.*",
    ".*githubusercontent\\.com.*",
]
