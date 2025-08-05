r"""
Fundamentals of TorchANI
========================

The TorchANI library provides a set of pre-trained models that can be used directly. All
provided models are subclasses of class `torchani.arch.ANI`.

It also contains classes such as `torchani.aev.AEVComputer`, `torchani.nn.ANINetworks`,
and `torchani.sae.SelfEnergy` that can be combined to compute molecular energies
from the cartesian coordinates of molecules, and it includes tools to deal with ANI
datasets in the submodule `torchani.datasets`.

Here introduce the basics of the library, and we show how to use the built-in models to
useful quantities of single molecules and batches.
"""

# To begin with, let's first import the modules we will use:
import torch

import torchani
from torchani.grad import energies_and_forces

###############################################################################
# ``torchani`` dependens on the ``torch`` library. Some familiarity with it is helpful,
# but you don't need to be an expert with ``torch`` in order to use ``torchani``. The
# main object in ``torch`` is a ``Tensor``, which is very similar to a ``numpy``
# multi-dimensional array. Its three most important properties are *data type*
# (``dtype``), *device*, and *shape*. Each time you create one, you can pass arguments
# to specify these. Lets create a couple as an example

device = "cpu"

# The ``dtype`` of this tensor is ``torch.int64``, and its shape is (1, 5)
atomic_nums = torch.tensor([[6, 1, 1, 1, 1]], device=device)

# The ``dtype`` of this one is ``torch.float32``, its shape is (1, 5, 3)
coords = torch.tensor(
    [[[0.03192167, 0.00638559, 0.01301679],
      [-0.83140486, 0.39370209, -0.26395324],
      [-0.66518241, -0.84461308, 0.20759389],
      [0.45554739, 0.54289633, 0.81170881],
      [0.66091919, -0.16799635, -0.91037834]]],
    device=device,
)

###############################################################################
# These tensors represent the coordinates (in angstrom) and atomic_nums of a methane
# molecule. We put the tensors in ``"cpu"``, but you can use ``device="cuda"`` if you
# have a CUDA capable GPU. Tensors of coordinates in ``torchani`` have shape
# ``(molecules, atoms, 3)`` and dtype ``torch.float32|64``, tensors of atomic numbers
# have shape ``(molecules, atoms)``, and dtype ``torch.long``.
#
# ``torchani`` provides a set of pre-trained models, which are predictors that calculate
# properties of molecular systems. The built-in models can be used directly ``ANI``.
#
# Lets load the ANI-2x model, which is an 'ensemble' of 8 sub-models trained with
# different initial seeds, and to different sections of the "ANI-2x dataset". Using an
# ensemble is recommended, since althought its a bit more costly than using a single
# model, its significantly more accurate.

model = torchani.models.ANI2x(device=device)

###############################################################################
# As you can see, similarly to tensors, ``torchani`` models also have a ``device`` and a
# ``dtype``, which are the device and dtype their internal parameters. If your inputs
# are in CPU, then the model should be in CPU, and similarly for CUDA GPUs. By
# default all models have ``dtype=torch.float32``.

###############################################################################
# Now let's compute energy and force:
energy, force = energies_and_forces(model, atomic_nums, coords)

###############################################################################
# And print to see the result:
print("Energy (Hartree):", energy.item())
print("Force (Hartree / Å): \n", force)

###############################################################################
# you can get the atomic energies (WARNING: these have no physical meaning)
# by calling:
_, atomic_energies = model.atomic_energies((atomic_nums, coords))

###############################################################################
# this gives you the average (shifted) energies over all models of the ensemble
# by default, with the same shape as the coordinates.
# (Dummy atoms, if present, will have an energy of zero.)
print("Average Atomic energies: \n", atomic_energies)

###############################################################################
# you can access model specific atomic energies too:
_, atomic_energies = model.atomic_energies((atomic_nums, coords), ensemble_values=True)
print("Atomic energies of first model: \n", atomic_energies[0, :, :])
