# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import torch
from .nn import Sequential
import ase.neighborlist
from . import utils
import ase.calculators.calculator
import ase.units
import copy


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
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, species, aev_computer, model, energy_shifter, dtype=torch.float64, overwrite=False):
        super(Calculator, self).__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        # aev_computer.neighborlist will be changed later, so we need a copy to
        # make sure we do not change the original object
        aev_computer = copy.deepcopy(aev_computer)
        self.aev_computer = aev_computer.to(dtype)
        self.model = copy.deepcopy(model)
        self.energy_shifter = copy.deepcopy(energy_shifter)
        self.overwrite = overwrite

        self.device = self.aev_computer.EtaR.device
        self.dtype = dtype

        self.nn = Sequential(
            self.model,
            self.energy_shifter
        ).to(dtype)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()
        species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)
        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled:
            coordinates = utils.map2central(cell, coordinates, pbc)
            if self.overwrite and atoms is not None:
                atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)

        if pbc_enabled:
            if 'stress' in properties:
                cell = cell @ scaling
            aev = self.aev_computer((species, coordinates), cell=cell, pbc=pbc).aevs
        else:
            aev = self.aev_computer((species, coordinates)).aevs

        energy = self.nn((species, aev)).energies
        energy *= ase.units.Hartree
        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0]
            self.results['forces'] = forces.squeeze().to('cpu').numpy()

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()
