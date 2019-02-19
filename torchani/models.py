# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has two models: ANI-1x and ANI-1ccx. The corresponding
classes of these two models are :class:`ANI1x` and :class:`ANI1ccx`. These
classes share the same API. To use the builtin models, you simply need to
create an object of its corresponding class. These classes are subclasses of
:class:`torch.nn.Module`, and could be used directly. Below is an example of
how to use these models:

.. code:: python

    model = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = model((species, coordinates))
    model.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    model.species_to_tensor('CHHHH')

    model0 = model[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor('CHHHH')
"""

import torch
from . import neurochem


class BuiltinModels(torch.nn.Module):

    def __init__(self, builtin_class):
        super(BuiltinModels, self).__init__()
        self.builtins = builtin_class()
        self.aev_computer = self.builtins.aev_computer
        self.neural_networks = self.builtins.models
        self.energy_shifter = self.builtins.energy_shifter
        self.species_to_tensor = self.builtins.consts.species_to_tensor

    def forward(self, species_coordinates):
        species_aevs = self.aev_computer(species_coordinates)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    def __getitem__(self, index):
        ret = torch.nn.Sequential(
            self.aev_computer,
            self.neural_networks[index],
            self.energy_shifter
        )

        def ase():
            from . import ase
            return ase.Calculator(self.builtins.species,
                                  self.aev_computer,
                                  self.neural_networks[index],
                                  self.energy_shifter)

        ret.ase = ase
        ret.species_to_tensor = self.builtins.consts.species_to_tensor
        return ret

    def __len__(self):
        return len(self.neural_networks)

    def ase(self):
        """Get an ASE Calculator using this model"""
        from . import ase
        return ase.Calculator(self.builtins.species, self.aev_computer,
                              self.neural_networks, self.energy_shifter)


class ANI1x(BuiltinModels):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and
    `Active Learning Paper`_.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """

    def __init__(self):
        super(ANI1x, self).__init__(neurochem.Builtins)


class ANI1ccx(BuiltinModels):
    """The ANI-1x model as in `ani-1ccx_8x on GitHub`_ and
    `Transfer Learning Paper`_.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """

    def __init__(self):
        super(ANI1ccx, self).__init__(neurochem.BuiltinsANI1CCX)
