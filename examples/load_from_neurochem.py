# -*- coding: utf-8 -*-
"""
Construct Model From NeuroChem Files
====================================

This tutorial illustrates how to manually load model from `NeuroChem files`_.

.. _NeuroChem files:
    https://github.com/isayev/ASE_ANI/tree/master/ani_models

"""

###############################################################################
# To begin with, let's first import the modules we will use:
import os
import torch
import torchani
import ase


###############################################################################
# Now let's read constants from constant file and construct AEV computer.
try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
const_file = os.path.join(path, '../torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
consts = torchani.neurochem.Constants(const_file)
aev_computer = torchani.AEVComputer(**consts)

###############################################################################
# Now let's read self energies and construct energy shifter.
sae_file = os.path.join(path, '../torchani/resources/ani-1x_8x/sae_linfit.dat')  # noqa: E501
energy_shifter = torchani.neurochem.load_sae(sae_file)

###############################################################################
# Now let's read a whole ensemble of models.
model_prefix = os.path.join(path, '../torchani/resources/ani-1x_8x/train')  # noqa: E501
ensemble = torchani.neurochem.load_model_ensemble(consts.species, model_prefix, 8)  # noqa: E501

###############################################################################
# Or alternatively a single model.
model_dir = os.path.join(path, '../torchani/resources/ani-1x_8x/train0/networks')  # noqa: E501
model = torchani.neurochem.load_model(consts.species, model_dir)

###############################################################################
# You can create the pipeline of computing energies:
# (Coordinates) -[AEVComputer]-> (AEV) -[Neural Network]->
# (Raw energies) -[EnergyShifter]-> (Final energies)
# From using either the ensemble or a single model:
nnp1 = torchani.nn.Sequential(aev_computer, ensemble, energy_shifter)
nnp2 = torchani.nn.Sequential(aev_computer, model, energy_shifter)
print(nnp1)
print(nnp2)

###############################################################################
# You can also create an ASE calculator using the ensemble or single model:
calculator1 = torchani.ase.Calculator(consts.species, nnp1)
calculator2 = torchani.ase.Calculator(consts.species, nnp2)
print(calculator1)
print(calculator1)

###############################################################################
# Now let's define a methane molecule
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True)
species = consts.species_to_tensor(['C', 'H', 'H', 'H', 'H']).unsqueeze(0)
methane = ase.Atoms(['C', 'H', 'H', 'H', 'H'], positions=coordinates.squeeze().detach().numpy())

###############################################################################
# Now let's compute energies using the ensemble directly:
energy = nnp1((species, coordinates)).energies
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative
print('Energy:', energy.item())
print('Force:', force.squeeze())

###############################################################################
# And using the ASE interface of the ensemble:
methane.calc = calculator1
print('Energy:', methane.get_potential_energy() / ase.units.Hartree)
print('Force:', methane.get_forces() / ase.units.Hartree)

###############################################################################
# We can do the same thing with the single model:
energy = nnp2((species, coordinates)).energies
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative
print('Energy:', energy.item())
print('Force:', force.squeeze())

methane.calc = calculator2
print('Energy:', methane.get_potential_energy() / ase.units.Hartree)
print('Force:', methane.get_forces() / ase.units.Hartree)
