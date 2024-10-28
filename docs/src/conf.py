import torchani

project = "TorchANI"
copyright = "2024, Roitberg Group"
author = "Xiang Gao"

version = torchani.__version__
release = torchani.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]
autodoc_typehints_format = "short"  # Avoid qualified names in return types
autodoc_typehints = "description"  # Write types in description, not in signature
napoleon_google_docstring = True
napoleon_numpy_docstring = False
python_use_unqualifierd_type_names = True  # Not sure if needed

templates_path = ["_templates"]

source_suffix = {'.rst': 'restructuredtext'}
master_doc = "index"
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "TorchANIdoc"

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "examples_autogen",
    "filename_pattern": r".*\.py",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

latex_documents = [
    (master_doc, "TorchANI.tex", "TorchANI Documentation", "Xiang Gao", "manual"),
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
