import torchani

# General variables used in |substitutions|
project = "TorchANI"
copyright = "2024, Roitberg Group"
author = "TorchANI developers"
_version = ".".join(torchani.__version__.split(".")[:2])
version = f"{_version} (dev)" if "dev" in torchani.__version__ else _version
release = torchani.__version__

# Common substitutions used throughout the docs
rst_epilog = """
..  |coords| replace:: A float tensor with the coordinates of a batch of molecules.
    Shape is ``(molecules, atoms, 3)``. All ANI models use Angstrom.

..  |symbols| replace:: A `tuple` or `list` of strings that are valid chemical symbols.
    (case sensitive).

..  |atomic_nums| replace:: An int tensor that stores the atomic numbers of a batch of
    molecules. Shape is ``(molecules, 3)``.

..  |elem_idxs| replace:: An int `torch.Tensor` that stores the element indices of a
    batch of molecules, (for example after conversion with
    `torchani.nn.SpeciesConverter`). Shape is ``(molecules, 3)``.

..  |pbc| replace:: A bool tensor that stores whether periodic boundary conditions (PBC)
    are enabled for the x, y, z directions. ``pbc=torch.tensor([True, True, True])``
    fully enables PBC and ``pbc=None`` (or ``torch.tensor([False, False, False])``).
    fully disables it.

..  |neighbors| replace:: `typing.NamedTuple` that contains the output of a
    `torchani.neighbors.Neighborlist` calculation.

..  |distances| replace:: A float tensor with the distances between pairs of atoms in a
    system. Shape is ``(pairs,)``.

..  |cell| replace:: A float tensor with unit cell vectors in its *rows*. Only
    use this with PBC. A cell with dimensions 10, 15, 20 (in Angstrom for all ANI
    models) in the x, y, and z directions is given by ``torch.tensor([[10., 0., 0.],[0.,
    15., 0.],[0., 0., 20.]])``.

..  |aevs| replace:: A float tensor with local atomic features (AEVs). Shape is
    ``(molecules, atoms, num-aev-features)``.
"""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # For autogen python module docs
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # For google-style docstr
    "sphinx_gallery.gen_gallery",  # For rendering user guide
    "sphinx_design",  # For grid directive
]
# Autosummary
autosummary_ignore_module_all = False  # Respect <module>.__all__
# Extensions config
# autodoc
autodoc_typehints_format = "short"  # Avoid qualified names in autodoc types
autodoc_typehints = "description"  # Write types in description, not in signature
autodoc_typehints_description_target = "documented"  # Only write type for docum. params
autodoc_inherit_docstrings = True  # Docstring of supercls is used by dflt
autodoc_default_options = {
    "members": None,  # This means "True"
    "member-order": "bysource",  # Document in the same order as python source code
}
# napoleon
napoleon_google_docstring = True  # Use google-style docstrings only
napoleon_numpy_docstring = False
# sphinx-gallery
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "examples_autogen",
    "filename_pattern": r".*\.py",
    "show_signature": False,
    "min_reported_time": 3600,  # Don't report computation times
}
# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}


# General sphinx config
# nitpicky = True  # Fail if refs can't be resolved TODO re-enable and fix invalid refs
default_role = "any"  # Behavior of `inline-backticks`, try to link to "anything"
pygments_style = "sphinx"  # Code render style
templates_path = ["_templates"]
master_doc = "index"  # Default, Main toctree
source_suffix = {".rst": "restructuredtext"}  # Default, Suffix of files

# Python-domain sphinx config
python_use_unqualifierd_type_names = True  # Try to dequa py obj names if resolveable
python_display_short_literal_types = True  # show literals as a | b | ...

# HTML config
html_title = f"{project} v{version} Manual"
html_static_path = ["_static"]  # Static html resources
html_css_files = ["style.css"]  # Overrides for theme style sheet
html_theme = "pydata_sphinx_theme"
html_use_modindex = True
html_domain_indices = False
html_copy_source = False
html_file_suffix = ".html"
htmlhelp_basename = "torchani-docs"
# TODO
# html_logo = '_static/logo.svg'
# html_favicon = '_static/favicon.ico'

# PyData Theme config
# Primary HTML sidebar (left)
html_sidebars = {
    "index": [],
    "installing": [],
    "examples_autogen/*": ["sidebar-nav-bs"],
    "api_autogen/*": ["sidebar-nav-bs"],
    "publications": [],
}
html_theme_options = {
    # "show_nav_level": 1, (TODO: what does this do?)
    "show_toc_level": 1,  # default is 2?
    # "navigation_depth": 4,  Default (TODO what does this do?)
    "primary_sidebar_end": [],
    # navbar (Top bar)
    # "navbar_align": "content", Default
    # "navbar_start": ["navbar-logo"],  Default
    "navbar_center": ["navbar-nav"],
    # "navbar_persistent": ["search-button"], Default
    # "navbar_end": ["theme-switcher", "navbar-icon-links"], Default
    # "header_links_before_dropdown": 5,  # Default, Headers before collapse to "more v"
    # Secondary HTML sidebar (right)
    "secondary_sidebar_items": {
        "index": [],
        "installing": [],
        "examples_autogen/*": ["page-toc"],
        "api_autogen/*": ["page-toc"],
        "publications": [],
    },
    # Misc
    "github_url": "https://github.com/aiqm/torchani",
    "icon_links": [],
    "logo": {
        "image_light": "_static/torchani3-logo-light.svg",
        "image_dark": "_static/torchani3-logo-dark.svg",
    },
    # "logo": {"text": "TorchANI"},
    "show_version_warning_banner": True,
}

# Other: info, tex, man
latex_documents = [
    (
        master_doc,
        "TorchANI.tex",
        "TorchANI Documentation",
        "TorchANI developers",
        "manual",
    ),
]
man_pages = [(master_doc, "torchani", "TorchANI Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "TorchANI",
        "TorchANI Documentation",
        author,
        "TorchANI",
        "One line description of project.",
        "Miscellaneous",
    ),
]
