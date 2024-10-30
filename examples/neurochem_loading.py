r"""
Constructing a TorchANI model from NeuroChem files
==================================================

This tutorial illustrates how to manually load model from `NeuroChem files`_.

.. _NeuroChem files:
    https://github.com/isayev/ASE_ANI/tree/master/ani_models
"""
# To begin with, let's first import the modules we will use:
from pathlib import Path
import torch

from torchani.paths import neurochem_dir
from torchani.grad import energies_and_forces
from torchani.neurochem import (
    download_model_parameters,
    load_aev_computer_and_symbols,
    load_sae,
    load_member,
    load_ensemble,
    load_model_from_info_file,
)

###############################################################################
# First lets download all model parameters, by default they will be loaded into
# torchani.paths.NEUROCHEM, which is ~/.local/torchani/Neurochem.
root = neurochem_dir()
download_model_parameters()

###############################################################################
# Now let's read constants from constant file and construct AEV computer,
# The sae's and construct a SelfEnergy potential,
# and the networks to construct an ensemble
const_file = Path(root, "ani-1x_8x", "rHCNO-5.2R_16-3.5A_a4-8.params")
aev_computer, symbols = load_aev_computer_and_symbols(const_file)
model_prefix = Path(root, "ani-1x_8x", "train")
ensemble = load_ensemble(symbols, model_prefix, 8)
sae_file = Path(root, "ani-1x_8x", "sae_linfit.dat")
energy_shifter = load_sae(sae_file)

###############################################################################
# We can also load a single model from the ensemble
model_dir = Path(root, "ani-1x_8x", "train0", "networks")
member = load_member(symbols, model_dir)

###############################################################################
# We can also load an ANI model using neurochem
ensemble_model = load_model_from_info_file(root / "ani-1x_8x.info")
single_model = load_model_from_info_file(root / "ani-1x_8x.info", model_index=0)
print(ensemble_model)
print(single_model)

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
)
species = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.long)

###############################################################################
# Now let's compute energies using the ensemble directly:
energies, forces = energies_and_forces(ensemble_model, species, coordinates)
print("Energy:", energies.item())
print("Force:", forces.squeeze())

###############################################################################
# We can do the same thing with the single model:
energies, forces = energies_and_forces(single_model, species, coordinates)
print("Energy:", energies.item())
print("Force:", forces.squeeze())
