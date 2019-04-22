# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

from __future__ import absolute_import
import torch
import ase.neighborlist
from . import utils
import ase.calculators.calculator
import ase.units
import copy
import numpy


class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer.
        model (:class:`torchani.ANIModel` or :class:`torchani.Ensemble`):
            neural network potential models.
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter.
        dtype (:class:`torchani.EnergyShifter`): data type to use,
            by dafault ``torch.float64``.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, species, aev_computer, model, energy_shifter, dtype=torch.float64):
        super(Calculator, self).__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        # aev_computer.neighborlist will be changed later, so we need a copy to
        # make sure we do not change the original object
        self.aev_computer = copy.deepcopy(aev_computer)
        self.model = copy.deepcopy(model)
        self.energy_shifter = copy.deepcopy(energy_shifter)

        self.device = self.aev_computer.EtaR.device
        self.dtype = dtype

        self.whole = torch.nn.Sequential(
            self.aev_computer,
            self.model,
            self.energy_shifter
        ).to(dtype)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            requires_grad=True, dtype=self.dtype,
                            device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc().astype(numpy.uint8), dtype=torch.uint8,
                           device=self.device)
        # print(cell, pbc)
        species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)
        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.unsqueeze(0).to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)
        _, energy = self.whole((species, coordinates, cell, pbc))
        energy *= ase.units.Hartree
        self.results['energy'] = energy.item()
        if 'forces' in properties:
            forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0]
            self.results['forces'] = forces.squeeze().to('cpu').numpy()
