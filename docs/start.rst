Installation
============

TorchANI requires the latest preview version of PyTorch. You can install PyTorch by the following commands (assuming cuda10):

.. code-block:: bash

    pip install numpy torchvision_nightly
    pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html

If you updated TorchANI, you may also need to update PyTorch:

.. code-block:: bash

    pip install --upgrade torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html


After installing the correct PyTorch, you can install TorchANI by:

.. code-block:: bash

    pip install torchani

See also `PyTorch's official site`_ for instructions of installing latest preview version of PyTorch.

.. warning::

    Please install nightly PyTorch through ``pip install`` instead of ``conda install``. If your PyTorch is installed through ``conda install``, then `pip` would mistakenly recognize the package name as `torch` instead of `torch-nightly`, which would cause dependency issue when installing TorchANI.

.. _PyTorch's official site:
    https://pytorch.org/get-started/locally/
