"""Tools for interfacing with `ASE`_.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""
import warnings

import torch
import ase.calculators.calculator
import ase.units

from torchani import utils


class Calculator(ase.calculators.calculator.Calculator):
    """TorchANI calculator for ASE

    Arguments:
        model (:class:`torch.nn.Module`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in :class:`ase.Atoms`
            object with the wrapped positions.
        stress_partial_fdotr (bool): whether to use partial_fdotr approach to
            calculate stress. This approach does not need the cell's box
            information and could be used for multiple domians when running
            parallel on multi-GPUs using lammps. Default as False.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, model, overwrite: bool = False, stress_partial_fdotr: bool = False):
        super().__init__()
        self.model = model
        # Since ANI is used in inference mode, no gradients on model parameters are required here
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.overwrite = overwrite
        self.stress_partial_fdotr = stress_partial_fdotr

        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype

        try:
            periodic_table_index = model.periodic_table_index
        except AttributeError:
            periodic_table_index = False

        if not periodic_table_index:
            raise ValueError("ASE models must have periodic_table_index=True")

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = torch.tensor(self.atoms.get_cell(complete=True).array,
                            dtype=self.dtype, device=self.device)
        pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
                           device=self.device)
        pbc_enabled = pbc.any().item()

        species = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        species = species.unsqueeze(0)
        coordinates = torch.tensor(self.atoms.get_positions())
        coordinates = coordinates.to(self.device).to(self.dtype) \
                                 .requires_grad_('forces' in properties)

        if pbc_enabled and self.overwrite and atoms is not None:
            warnings.warn("""If overwrite is set for pbc calculations the cell list will be rebuilt every step,
                    also take into account you are loosing information this way""")
            coordinates = utils.map_to_central(coordinates, cell, pbc)
            atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())

        if 'stress' in properties and not self.stress_partial_fdotr:
            scaling = torch.eye(3, requires_grad=True, dtype=self.dtype, device=self.device)
            coordinates = coordinates @ scaling
        coordinates = coordinates.unsqueeze(0)

        if pbc_enabled:
            if 'stress' in properties and not self.stress_partial_fdotr:
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
            if self.stress_partial_fdotr:
                diff_vectors = self.model.aev_computer.neighborlist.get_diff_vectors()
                stress = self._get_stress_partial_fdotr(diff_vectors, energy, volume)
            else:
                stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
            self.results['stress'] = stress.detach().cpu().numpy()

    def _get_ani_forces(self, coordinates, energy, properties):
        return -torch.autograd.grad(energy.squeeze(), coordinates, retain_graph='stress' in properties)[0]

    @staticmethod
    def _get_stress_partial_fdotr(diff_vectors, energy, volume):
        dEdR = torch.autograd.grad(energy.squeeze(), diff_vectors, retain_graph=True)[0]
        virial = dEdR.transpose(0, 1) @ diff_vectors
        stress = virial / volume
        return stress

    @staticmethod
    # TODO figure out a way to test
    def _get_stress_forces_partial_fdotr(coordinates, diff_vectors, energy, volume):
        forces, dEdR = torch.autograd.grad(energy.squeeze(), [coordinates, diff_vectors], retain_graph=True)
        forces = torch.neg(forces)

        virial = dEdR.transpose(0, 1) @ diff_vectors
        stress = virial / volume
        return forces, stress
