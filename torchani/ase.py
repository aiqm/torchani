# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import math
import torch
import ase.neighborlist
from . import utils
import ase.calculators.calculator
import ase.units


class NeighborList:
    """ASE neighborlist computer

    Arguments:
        cell: same as in :class:`ase.Atoms`
        pbc: same as in :class:`ase.Atoms`
    """

    def __init__(self, cell=None, pbc=None):
        self.pbc = pbc
        self.cell = cell

    def __call__(self, species, coordinates, cutoff):
        conformations = species.shape[0]
        max_atoms = species.shape[1]
        neighbor_species = []
        neighbor_distances = []
        neighbor_vecs = []
        for i in range(conformations):
            s = species[i].unsqueeze(0)
            c = coordinates[i].unsqueeze(0)
            s, c = utils.strip_redundant_padding(s, c)
            s = s.squeeze()
            c = c.squeeze()
            atoms = s.shape[0]
            atoms_object = ase.Atoms(
                'C'*atoms,  # chemical symbols are not important here
                positions=c.numpy(),
                pbc=self.pbc,
                cell=self.cell)
            idx1, idx2, d, D = ase.neighborlist.neighbor_list(
                'ijdD', atoms_object, cutoff)
            idx1 = torch.from_numpy(idx1).to(coordinates.device)
            idx2 = torch.from_numpy(idx2).to(coordinates.device)
            d = torch.from_numpy(d).to(coordinates.device) \
                .to(coordinates.dtype)
            D = torch.from_numpy(D).to(coordinates.device) \
                .to(coordinates.dtype)
            neighbor_species1 = []
            neighbor_distances1 = []
            neighbor_vecs1 = []
            for i in range(atoms):
                this_atom_indices = (idx1 == i).nonzero().flatten()
                neighbor_indices = idx2[this_atom_indices]
                neighbor_species1.append(s[neighbor_indices])
                neighbor_distances1.append(d[this_atom_indices])
                neighbor_vecs1.append(D.index_select(0, this_atom_indices))
            for i in range(max_atoms - atoms):
                neighbor_species1.append(torch.full((1,), -1))
                neighbor_distances1.append(torch.full((1,), math.inf))
                neighbor_vecs1.append(torch.full((1, 3), 0))
            neighbor_species1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_species1, padding_value=-1)
            neighbor_distances1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_distances1, padding_value=math.inf)
            neighbor_vecs1 = torch.nn.utils.rnn.pad_sequence(
                neighbor_vecs1, padding_value=0)
            neighbor_species.append(neighbor_species1)
            neighbor_distances.append(neighbor_distances1)
            neighbor_vecs.append(neighbor_vecs1)
        neighbor_species = torch.nn.utils.rnn.pad_sequence(
            neighbor_species, batch_first=True, padding_value=-1)
        neighbor_distances = torch.nn.utils.rnn.pad_sequence(
            neighbor_distances, batch_first=True, padding_value=math.inf)
        neighbor_vecs = torch.nn.utils.rnn.pad_sequence(
            neighbor_vecs, batch_first=True, padding_value=0)
        return neighbor_species.permute(0, 2, 1), \
            neighbor_distances.permute(0, 2, 1), \
            neighbor_vecs.permute(0, 2, 1, 3)


class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer.
        model (:class:`torchani.ANIModel` or :class:`torchani.Ensemble`):
            neural network potential models.
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter.
    """

    def __init__(self, species, aev_computer, model, energy_shifter):
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        self.aev_computer = aev_computer
        self.model = model
        self.energy_shifter = energy_shifter

        self.device = self.aev_computer.EtaR.device
        self.dtype = self.aev_computer.EtaR.dtype

        self.whole = torch.nn.Sequential(
            self.aev_computer,
            self.model,
            self.energy_shifter
        )

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super(Calculator, self).calculate(atoms, properties, system_changes)
        self.aev_computer.neighbor_list = NeighborList(
            cell=self.atoms.get_cell(), pbc=self.atoms.get_pbc())
        species = self.species_to_tensor(self.atoms.get_chemical_symbols())
        coordinates = self.atoms.get_positions(wrap=True).unsqueeze(0)
        coordinates = torch.tensor(coordinates,
                                   device=self.device,
                                   dtype=self.dtype,
                                   requires_grad=('forces' in properties))
        _, energy = self.whole((species, coordinates)) * ase.units.Hartree
        self.results['energy'] = energy.item()
        if 'forces' in properties:
            forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0]
            self.results['forces'] = forces.item()
