"""
Constructing a Model From NeuroChem Files
=========================================

This tutorial illustrates how to manually load model from `NeuroChem files`_.

.. _NeuroChem files:
    https://github.com/isayev/ASE_ANI/tree/master/ani_models
"""
# To begin with, let's first import the modules we will use:
import torch

import torchani
from torchani.utils import ChemicalSymbolsToInts
from torchani.neurochem import (
    download_model_parameters,
    load_aev_computer_and_symbols,
    load_sae,
    load_model_ensemble,
    load_model,
)

###############################################################################
# First lets download all model parameters, by default they will be loaded into
# torchani.storage.NEUROCHEM_DIR, which is ~/.local/torchani/Neurochem.
root = torchani.storage.NEUROCHEM_DIR
download_model_parameters()

###############################################################################
# Now let's read constants from constant file and construct AEV computer,
# The sae's and construct an EnergyShifter,
# and the networks to construct an ensemble
const_file = root / "ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params"
aev_computer, symbols = load_aev_computer_and_symbols(const_file)

model_prefix = root / "ani-1x_8x/train"
ensemble = load_model_ensemble(symbols, model_prefix, 8)
sae_file = root / "ani-1x_8x/sae_linfit.dat"
energy_shifter = load_sae(sae_file)

###############################################################################
# We can also load a single model from the ensemble
model_dir = root / "ani-1x_8x/train0/networks"
model = load_model(symbols, model_dir)

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
# Now let's define a methane molecule
coordinates = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ],
    requires_grad=True,
)
species_to_tensor = ChemicalSymbolsToInts(symbols)
species = species_to_tensor(["C", "H", "H", "H", "H"]).unsqueeze(0)

###############################################################################
# Now let's compute energies using the ensemble directly:
energy = nnp1((species, coordinates)).energies
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative
print("Energy:", energy.item())
print("Force:", force.squeeze())

###############################################################################
# We can do the same thing with the single model:
energy = nnp2((species, coordinates)).energies
derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
force = -derivative
print("Energy:", energy.item())
print("Force:", force.squeeze())
