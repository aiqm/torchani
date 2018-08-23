import torchani  # noqa: F401
import sphinx_rtd_theme

project = 'TorchANI'
copyright = '2018, Roitberg Group'
author = 'Xiang Gao'

version = '0.1'
release = '0.1alpha'

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
html_static_path = ['_static']

source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'TorchANIdoc'

latex_documents = [
    (master_doc, 'TorchANI.tex', 'TorchANI Documentation',
     'Xiang Gao', 'manual'),
]

man_pages = [
    (master_doc, 'torchani', 'TorchANI Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'TorchANI', 'TorchANI Documentation',
     author, 'TorchANI', 'One line description of project.',
     'Miscellaneous'),
]
