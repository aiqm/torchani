r"""Calculator subclass, used for interfacing with the ASE library

The `ASE`_ library can be used to perform molecular dynamics. For more information
consult the user guide.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import typing as tp
import warnings

import torch
from torch import Tensor

try:
    import ase.units
    from ase.calculators.calculator import Calculator as AseCalculator, all_changes
except ImportError:
    raise ImportError(
        "Error when trying to import 'torchani.ase':"
        " The ASE package could not be found. 'torchani.ase' and the '*.ase()' methods"
        " of models won't be available. Please install ase if you want to use them."
        " ('conda install ase' or 'pip install ase')"
    ) from None

from torchani.annotations import StressKind
from torchani.neighbors import Neighbors
from torchani.utils import map_to_central


class Calculator(AseCalculator):
    """TorchANI calculator for ASE

    ANI models can be converted to their ASE Calculator form by calling the
    ``ANI.ase`` method.

    .. code-block:: python

        import torchani
        model = torchani.models.ANI1x()
        calc = model.ase()  # Convert model into its ASE Calculator form

    Arguments:
        model (`torchani.arch.ANI`): neural network potential model
            that convert coordinates into energies.
        overwrite (bool): After wrapping atoms into central box, whether
            to replace the original positions stored in `ase.Atoms`
            object with the wrapped positions.
        stress_kind (str): Strategy to calculate stress, valid options are *fdotr*,
            *scaling*, and *numerical*. The fdotr approach does not need the cell's box
            information and can be used for multiple domians when running parallel on
            multi-GPUs.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model,
        overwrite: bool = False,
        stress_kind: StressKind = "scaling",
    ):
        super().__init__()
        self.model = model
        param = next(self.model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        if not model.periodic_table_index:
            raise ValueError("ASE models must have periodic_table_index=True")

        self.overwrite = overwrite
        self.stress_kind = stress_kind

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None  # mypy
        needs_stress = "stress" in properties
        needs_forces = "forces" in properties
        species = torch.tensor(
            self.atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        coords = torch.tensor(
            self.atoms.get_positions(),
            device=self.device,
            dtype=self.dtype,
            requires_grad=needs_forces,
        )
        cell: tp.Optional[Tensor] = torch.tensor(
            self.atoms.get_cell(complete=True).array,
            dtype=self.dtype,
            device=self.device,
        )
        pbc: tp.Optional[Tensor] = torch.tensor(
            self.atoms.get_pbc(), dtype=torch.bool, device=self.device
        )
        assert pbc is not None
        assert cell is not None

        if pbc.any() and self.overwrite:
            warnings.warn("'overwrite' set, info about crossing PBC *will be lost*")
            coords = map_to_central(coords, cell, pbc)
            self.atoms.set_positions(coords.detach().cpu().reshape(-1, 3).numpy())

        if not pbc.any():
            cell = None
            pbc = None

        if needs_stress and self.stress_kind == "scaling":
            scaling = torch.eye(
                3, requires_grad=True, dtype=self.dtype, device=self.device
            )
            coords = coords @ scaling
        coords = coords.unsqueeze(0)

        if needs_stress:
            elem_idxs = self.model.species_converter(species)
            if self.stress_kind == "scaling":
                cell = cell @ scaling
            neighbors = self.model.neighborlist(
                self.model.cutoff, species, coords, cell, pbc
            )
            if self.stress_kind == "fdotr":
                neighbors.diff_vectors.requires_grad_(True)
                neighbors = Neighbors(
                    neighbors.indices,
                    neighbors.diff_vectors.norm(2, -1),
                    neighbors.diff_vectors,
                )
            result = self.model.compute_from_neighbors(elem_idxs, coords, neighbors)
            energy = result.energies
        else:
            energy = self.model((species, coords), cell, pbc).energies
        energy = energy * ase.units.Hartree
        self.results["energy"] = energy.item()

        # Unclear what free_energy means in ASE, it is returned from
        # get_potential_energy(force_consistent=True), and required for
        # calculate_numerical_stress
        self.results["free_energy"] = energy.item()
        if needs_forces:
            self.results["forces"] = self._forces(coords, energy, retain=needs_stress)
        if needs_stress:
            volume = self.atoms.get_volume()
            if self.stress_kind == "fdotr":
                # Neighbors must be not-none if stress is needed
                self.results["stress"] = self._stress_fdotr(
                    neighbors.diff_vectors, energy, volume
                )
            elif self.stress_kind == "numerical":
                self.results["stress"] = self.calculate_numerical_stress(self.atoms)
            elif self.stress_kind == "scaling":
                self.results["stress"] = self._stress_scaling(energy, scaling, volume)
            else:
                raise ValueError(f"Unsupported stress kind {self.stress_kind}")

    @staticmethod
    def _forces(coords, energy, retain):
        forces = -torch.autograd.grad(energy.squeeze(), coords, retain_graph=retain)[0]
        return forces.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def _stress_fdotr(diff_vectors, energy, volume):
        dEdR = torch.autograd.grad(energy.squeeze(), diff_vectors)[0]
        virial = dEdR.transpose(0, 1) @ diff_vectors
        return (virial / volume).detach().cpu().numpy()

    @staticmethod
    def _stress_scaling(energy, scaling, volume):
        stress = torch.autograd.grad(energy.squeeze(), scaling)[0] / volume
        return stress.detach().cpu().numpy()
