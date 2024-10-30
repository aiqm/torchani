r"""
Using TorchScript to serialize models
=====================================

All built-in models and modules in the `torchani` library, support TorchScript
serialization, which is a native PyTorch feature where a python model is translated into
a PyTorch-specific format. If you use TorchScript you can load the resulting serialized
files in a process where there is no Python dependency.
"""
# To begin with, let's first import the modules we will use:
from pathlib import Path
import typing as tp

import torch
from torch import Tensor

import torchani
from torchani.grad import hessians, forces

###############################################################################
# Scripting an ANI model directly
# --------------------------------
#
# Let's now load the built-in ANI-1ccx models. The ANI-2x model contains 8
# models trained with diffrent initialization and on different splits of a dataset
model = torchani.models.ANI2x()

###############################################################################
# It is very easy to compile and save the model using `torch.jit`.
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, "compiled_model.pt")

###############################################################################
# For testing purposes, we will now load the model we just saved and see if
# they produces the same output as the original model:
loaded_compiled_model = torch.jit.load("compiled_model.pt")


###############################################################################
# We use the molecule below to test:
coordinates = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ]
)
# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]])

###############################################################################
# And here is the result:
energies_ensemble = model((species, coordinates)).energies
energies_ensemble_jit = loaded_compiled_model((species, coordinates)).energies
print(
    "Ensemble energy, eager mode vs loaded jit:",
    energies_ensemble.item(),
    energies_ensemble_jit.item(),
)


###############################################################################
# Customize the model and script
# ------------------------------
#
# You could also customize the model you want to export. For example, let's do
# the following customization to the model:
#
# - uses double as dtype instead of float
# - don't care about periodic boundary condition
# - in addition to energies, allow returning optionally forces, and hessians
#
# you could do the following:
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchani.models.ANI1x().double()

    def forward(
        self,
        species: Tensor,
        coordinates: Tensor,
        return_forces: bool = False,
        return_hessians: bool = False,
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor], tp.Optional[Tensor]]:
        if return_forces or return_hessians:
            coordinates.requires_grad_(True)
        energies = self.model((species, coordinates)).energies
        _forces: tp.Optional[Tensor] = None
        _hessians: tp.Optional[Tensor] = None
        if return_forces or return_hessians:
            _forces = forces(
                energies, coordinates, retain_graph=True, create_graph=return_hessians
            )
            if return_hessians:
                assert _forces is not None
                _hessians = hessians(_forces, coordinates)
        return energies, _forces, _hessians


custom_model = CustomModule()
compiled_custom_model = torch.jit.script(custom_model)
torch.jit.save(compiled_custom_model, "compiled_custom_model.pt")
loaded_compiled_custom_model = torch.jit.load("compiled_custom_model.pt")
energies_eager, forces_eager, hessians_eager = custom_model(
    species, coordinates, True, True
)
energies_jit, forces_jit, hessians_jit = loaded_compiled_custom_model(
    species, coordinates, True, True
)

print("Energy, eager mode vs loaded jit:", energies_eager.item(), energies_jit.item())
print()
print(
    "Force, eager mode vs loaded jit:\n",
    forces_eager.squeeze(0),
    "\n",
    forces_jit.squeeze(0),
)
print()
torch.set_printoptions(sci_mode=False, linewidth=1000)
print(
    "Hessian, eager mode vs loaded jit:\n",
    hessians_eager.squeeze(0),
    "\n",
    hessians_jit.squeeze(0),
)
# Lets delete the files we created for cleanup
Path("compiled_custom_model.pt").unlink()
Path("compiled_model.pt").unlink()
