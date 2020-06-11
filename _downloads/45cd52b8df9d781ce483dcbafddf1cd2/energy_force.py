# -*- coding: utf-8 -*-
"""
Computing Energy and Force Using Models Inside Model Zoo
========================================================

TorchANI has a model zoo trained by NeuroChem. These models are shipped with
TorchANI and can be used directly.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torch
import torchani

###############################################################################
# Let's now manually specify the device we want TorchANI to run:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Let's now load the built-in ANI-1ccx models. The builtin ANI-1ccx contains 8
# models trained with diffrent initialization. Predicting the energy and force
# using the average of the 8 models outperform using a single model, so it is
# always recommended to use an ensemble, unless the speed of computation is an
# issue in your application.
#
# The ``periodic_table_index`` arguments tells TorchANI to use element index
# in periodic table to index species. If not specified, you need to use
# 0, 1, 2, 3, ... to index species
model = torchani.models.ANI2x(periodic_table_index=True).to(device)

###############################################################################
# Now let's define the coordinate and species. If you just want to compute the
# energy and force for a single structure like in this example, you need to
# make the coordinate tensor has shape ``(1, Na, 3)`` and species has shape
# ``(1, Na)``, where ``Na`` is the number of atoms in the molecule, the
# preceding ``1`` in the shape is here to support batch processing like in
# training. If you have ``N`` different structures to compute, then make it
# ``N``.
#
# .. note:: The coordinates are in Angstrom, and the energies you get are in Hartree
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)

###############################################################################
# Now let's compute energy and force:
energy = model((species, coordinates)).energies
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative

###############################################################################
# And print to see the result:
print('Energy:', energy.item())
print('Force:', force.squeeze())
