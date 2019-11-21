# -*- coding: utf-8 -*-
"""
Using TorchScript to serialize and deploy model
===============================================

Models in TorchANI's model zoo support TorchScript. TorchScript is a way to create
serializable and optimizable models from PyTorch code. It allows users to saved their
models from a Python process and loaded in a process where there is no Python dependency.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torch
import torchani

###############################################################################
# Let's now load the built-in ANI-1ccx models. The builtin ANI-1ccx contains 8
# models trained with diffrent initialization.
model = torchani.models.ANI1ccx(periodic_table_index=True)

###############################################################################
# It is very easy to compile and save the model using `torch.jit`.
compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, 'compiled_model.pt')

###############################################################################
# Besides compiling the ensemble, it is also possible to compile a single network
compiled_model0 = torch.jit.script(model[0])
torch.jit.save(compiled_model0, 'compiled_model0.pt')

###############################################################################
# For testing purposes, we will now load the models we just saved and see if they
# produces the same output as the original model:
loaded_compiled_model = torch.jit.load('compiled_model.pt')
loaded_compiled_model0 = torch.jit.load('compiled_model0.pt')


###############################################################################
# We use the molecule below to test:
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]])
# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]])

###############################################################################
# And here is the result:
energies_ensemble = model((species, coordinates)).energies
energies_single = model[0]((species, coordinates)).energies
energies_ensemble_jit = loaded_compiled_model((species, coordinates)).energies
energies_single_jit = loaded_compiled_model0((species, coordinates)).energies
print('Ensemble energy, eager mode vs loaded jit:', energies_ensemble.item(), energies_ensemble_jit.item())
print('Single network energy, eager mode vs loaded jit:', energies_single.item(), energies_single_jit.item())
