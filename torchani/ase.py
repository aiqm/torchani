# -*- coding: utf-8 -*-
"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import torch
from . import utils
import ase.calculators.calculator
import ase.units
import warnings


class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
            sequence of all supported species, in order.
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, species, model, overwrite=False):
        super().__init__()
        self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
        self.model = model
        # Since ANI is used in inference mode, no gradients on model parameters are required here
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.overwrite = overwrite

        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        try:
            # We assume that the model has a "periodic_table_index" attribute
            # if it doesn't we set the calculator's attribute to false and we
            # assume that species will be correctly transformed by
            # species_to_tensor
            self.periodic_table_index = model.periodic_table_index
        except AttributeError:
            self.periodic_table_index = False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True),
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()

        if self.periodic_table_index:
            species = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        else:
            species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)

        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled and self.overwrite and atoms is not None:
            warnings.warn("""If overwrite is set for pbc calculations the cell list will be rebuilt every step,
                    also take into account you are loosing information this way""")
            coordinates = utils.map_to_central(coordinates, cell, pbc)
            atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)

        if pbc_enabled:
            if 'stress' in properties:
                cell = cell @ scaling
            energy = self.model((species, coordinates), cell=cell, pbc=pbc).energies
        else:
            energy = self.model((species, coordinates)).energies

        energy *= ase.units.Hartree
        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        if 'forces' in properties:
            forces = self._get_ani_forces(coordinates, energy, properties)
            self.results['forces'] = forces.squeeze(0).to('cpu').numpy()

        if 'stress' in properties:
            volume = self.atoms.get_volume()
            stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
            self.results['stress'] = stress.cpu().numpy()

    def _get_ani_forces(self, coordinates, energy, properties):
        return -torch.autograd.grad(energy.squeeze(), coordinates, retain_graph='stress' in properties)[0]
