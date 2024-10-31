:html_theme.sidebar_secondary.remove: true

.. This documentation is based on that of Pandas and SciPy, and uses the PyData theme

.. currentmodule:: torchani

**********************
TorchANI documentation
**********************

.. toctree::
   :hidden:

   installing
   user-guide
   API reference <api_autogen/torchani>
   publications

**Date**: |today| **Version**: |version|

**Useful links**:
`GitHub Repo <https://github.com/aiqm/torchani>`_  |
`Issue Tracker <https://github.com/aiqm/torchani/issues>`_  |
`Roitberg Lab <https://roitberg.chem.ufl.edu>`_

**TorchANI** is an open-source library that supports training, development, and research
of ANI-style neural network interatomic potentials. It was originally developed and is
currently maintained by the Roitberg group.

If you were using TorchANI before version 3, your code may need updating to be
compatible with its new features. Please consult :ref:`the migration guide
<torchani-migrating>` for more information. If you still have problems, please open an
issue on the GitHub `issue tracker <https://github.com/aiqm/torchani/issues>`_.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: _static/installing.svg
        :text-align: center

        **Installing**
        ^^^

        Want to install the TorchANI library? Read this section for details on how to
        install using ``conda`` or ``pip``. If you want to build TorchANI from source
        instead, please consult the ``README`` in the GitHub repo.

        +++

        .. button-ref:: installing
            :color: secondary
            :click-parent:

            To the installation guide

    .. grid-item-card::
        :img-top: _static/user-guide.svg
        :text-align: center

        **User guide**
        ^^^

        Start here! This sections describes how to use the TorchANI models in your
        research or application. It also teaches you about the main classes in in the
        library, and how to extend them.

        +++

        .. button-ref:: user-guide
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: _static/api.svg
        :text-align: center

        **API reference**
        ^^^

        A detailed description of the TorchANI API. It shows all its *public* functions,
        classes, and their methods and properties. Use this for reference. It assumes a
        basic understanding of Python and ``torch``.

        +++

        .. button-ref:: api_autogen/torchani
            :color: secondary
            :click-parent:

            To the API reference

    .. grid-item-card::
        :img-top: _static/publications.svg
        :text-align: center

        **Publications**
        ^^^

        Articles regarding TorchANI itself, and also with specific methods and models it
        implements. Please consult and cite the corresponding articles if you use
        TorchANI in a scientific publication.

        +++

        .. button-ref:: publications
            :color: secondary
            :click-parent:

            To the publications page
