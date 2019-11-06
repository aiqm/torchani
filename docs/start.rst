Installation
============

TorchANI requires the latest preview version of PyTorch. You can install PyTorch by the following commands (assuming cuda10):

.. code-block:: bash

    pip install numpy
    pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html

If you updated TorchANI, you may also need to update PyTorch:

.. code-block:: bash

    pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html


After installing the correct PyTorch, you can install TorchANI by:

.. code-block:: bash

    pip install torchani

See also `PyTorch's official site`_ for instructions of installing latest preview version of PyTorch.

.. _PyTorch's official site:
    https://pytorch.org/get-started/locally/
