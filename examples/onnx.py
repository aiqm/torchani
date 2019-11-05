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
model = torchani.models.ANI1ccx()

###############################################################################
# It is very easy to compile and save the model using `torch.jit`.
compiled_model = torch.jit.script(model)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]])
species = model.species_to_tensor('CHHHH').unsqueeze(0)
input_ = (species, coordinates)

torch.onnx.export(compiled_model, (input_,), 'ani1ccx.onnx', example_outputs=compiled_model(input_), verbose=True)
