# -*- coding: utf-8 -*-
"""
Computing Energy and Force Using Builtin Models
===============================================

TorchANI has a model ensemble trained by NeuroChem on the `ANI-1x dataset`_.
These models are shipped with TorchANI and can be used directly. To begin with,
let's first import the modules we will use:

.. _ANI-1x dataset:
  https://aip.scitation.org/doi/abs/10.1063/1.5023802
"""

import torch
import torchani

###############################################################################
# Let's now manually specify the device we want TorchANI to run:
device = torch.device('cpu')

###############################################################################
# Let's now load the built-in models and create a pipeline of AEV computer,
# neural networks, and energy shifter. This pipeline will first compute AEV,
# then use all models in the ensemble to compute molecular energies, and take
# the average of these energies to obtain a final output. The reason we need an
# energy shifter in the end is that the output of these networks is not the
# total energy but the total energy subtracted by a self energy for each atom.
builtin = torchani.neurochem.Builtins()
model = torch.nn.Sequential(
  builtin.aev_computer,
  builtin.models,
  builtin.energy_shifter
)

###############################################################################
# Now let's define the coordinate and species. If you just want to compute the
# energy and force for a single structure like in this example, you need to
# make the coordinate tensor has shape ``(1, Na, 3)`` and species has shape
# ``(1, Na)``, where ``Na`` is the number of atoms in the molecule, the
# preceding ``1`` in the shape is here to support batch processing like in
# training. If you have ``N`` different structures to compute, then make it
# ``N``.
coordinates = torch.tensor([[[0.03192167,  0.00638559,  0.01301679],
                             [-0.83140486,  0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308,  0.20759389],
                             [0.45554739,   0.54289633,  0.81170881],
                             [0.66091919,  -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device)
species = builtin.consts.species_to_tensor('CHHHH').to(device).unsqueeze(0)

###############################################################################
# Now let's compute energy and force:
_, energy = model((species, coordinates))
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative

###############################################################################
# And print to see the result:
print('Energy:', energy.item())
print('Force:', force.squeeze())
