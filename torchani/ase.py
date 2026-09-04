r"""Calculator subclass, used for interfacing with the ASE library

The `ASE`_ library can be used to perform molecular dynamics. For more information
consult the user guide.

.. _ASE:
    https://wiki.fysik.dtu.dk/ase
"""

import warnings

import torch
from torch import Tensor
import ase
import ase.units
import numpy as np
from numpy.typing import NDArray
from ase.calculators.calculator import (
    Calculator as AseCalculator,
    all_changes as _ALL_CHANGES,
)

from torchani.annotations import StressKind, Device, DType
from torchani.neighbors import Neighbors
from torchani.utils import map_to_central

_DEFAULT_PROPERTIES = ["energy"]


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
        self, model, overwrite: bool = False, stress_kind: StressKind = "scaling"
    ):
        super().__init__()
        if hasattr(model, "periodic_table_index") and not model.periodic_table_index:
            raise ValueError("ASE models must have periodic_table_index=True")
        param = next(model.parameters())
        self.model = model
        self._device = param.device
        self._dtype = param.dtype
        self._overwrite = overwrite
        self._stress_kind = stress_kind

    # NOTE: The ASE default is _ALL_CHANGES ==
    # ["positions", "numbers", "cell", "pbc", "initial_charges", "initial_magmoms"]
    # NOTE: Bad idea to use lists as defaults, but this is what ASE does
    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] = _DEFAULT_PROPERTIES,
        system_changes: list[str] = _ALL_CHANGES,
    ):
        # NOTE: If atoms is passed, then the
        # superclass overwrites self.atoms with the passed atoms
        super().calculate(atoms, properties, system_changes)
        # TODO: We should not need to reset the results every time this is called,
        # bugs associated with stress and force calculation should be investigated,
        # since this should be a no-op
        # Clear results, added by WardLT
        self.results: dict[str, NDArray[np.floating] | float] = {}
        if self.atoms is None:
            raise ValueError("Can't calculate if not attached to Atoms")

        species, coords, cell, pbc = from_ase(self.atoms, self._device, self._dtype)
        if "forces" in properties:
            coords.requires_grad_(True)

        scaling = torch.eye(3, dtype=self._dtype, device=self._device)
        if "stress" in properties and self._stress_kind == "scaling":
            scaling.requires_grad_(True)

        if self._overwrite and cell is not None:
            assert pbc is not None  # mypy
            warnings.warn("'overwrite' set, info about crossing PBC *will be lost*")
            coords = map_to_central(coords, cell, pbc)
            _set_atom_positions_from_tensor(self.atoms, coords)

        if "stress" in properties:
            if cell is None:
                raise ValueError("Can't require stress if not using PBC")
            assert pbc is not None  # mypy
            assert cell is not None  # mypy

            if self._stress_kind == "scaling":
                coords = coords @ scaling
                cell = cell @ scaling

            neighbors = self.model.neighborlist(
                self.model.cutoff, species, coords, cell, pbc
            )
            if self._stress_kind == "fdotr":
                diff_vec = neighbors.diff_vectors
                diff_vec.requires_grad_(True)
                neighbors = Neighbors(neighbors.indices, diff_vec.norm(2, -1), diff_vec)

            energy = self.model.compute_from_neighbors(
                self.model.species_converter(species), coords, neighbors
            ).energies
        else:
            energy = self.model((species, coords), cell, pbc).energies

        energy = energy * ase.units.Hartree
        if "energy" in properties:
            self._set_result("energy", energy)
        # NOTE: 'free energy' seems to mean smth slightly different in ASE context
        # it is what is get_potential_energy(force_consistent=True) returns
        if "free_energy" in properties:
            self._set_result("free_energy", energy)

        # [Coords, scaling/diff_vecs]
        if "forces" in properties or (
            "stress" in properties and self._stress_kind != "numerical"
        ):
            inputs = [self._dummy_tensor_req_grad(), self._dummy_tensor_req_grad()]
            if "forces" in properties:
                inputs[0] = coords
            if "stress" in properties:
                if self._stress_kind == "scaling":
                    inputs[1] = scaling
                elif self._stress_kind == "fdotr":
                    inputs[1] = diff_vec
            grad = torch.autograd.grad(energy.sum(), inputs, allow_unused=True)

        if "forces" in properties:
            self._set_result("forces", -grad[0])

        if "stress" in properties:
            volume = torch.linalg.det(cell).abs()
            if self._stress_kind == "numerical":
                virial = self._calc_numerical_virial(species, coords, cell)
            elif self._stress_kind == "scaling":
                virial = grad[1]
            elif self._stress_kind == "fdotr":
                virial = grad[1].transpose(0, 1) @ diff_vec  # grad[1] == dE/d(diff_vec)
            else:
                raise ValueError(f"Unsupported stress kind {self._stress_kind}")
            self._set_result("stress", virial / volume)

    def _set_result(self, key: str, value: Tensor) -> None:
        if value.ndim in (0, 1):
            self.results[key] = value.item()
        else:
            if value.ndim == 3:
                value = value.squeeze(0)
            self.results[key] = value.detach().cpu().numpy()

    # Calculate virial numerically based on the finite-difference method.
    # NOTE: Mostly copied from ASE calculators.fd code, but their codebase changes a
    # lot, so reproduced herefor consistency across ASE versions
    # Also this version is pure torch
    def _calc_numerical_virial(
        self,
        species: Tensor,
        coords: Tensor,
        cell: Tensor | None,
        eps: float = 1e-6,
    ) -> Tensor:
        if cell is None:
            raise ValueError("Cell is required for calculating numerical virial")
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=self._device)
        virial = torch.zeros((3, 3), dtype=self._dtype, device=self._device)
        for i in range(3):
            # Diagonal terms
            x = torch.eye(3, dtype=self._dtype, device=self._device)
            x[i, i] = 1.0 + eps
            eplus = self.model((species, coords @ x), cell @ x, pbc).energies
            x[i, i] = 1.0 - eps
            eminus = self.model((species, coords @ x), cell @ x, pbc).energies
            virial[i, i] = (eplus - eminus) / (2 * eps)

            # Off diagonal terms
            x = torch.eye(3, dtype=self._dtype, device=self._device)  # Reset
            j = i - 2
            x[i, j] = x[j, i] = +0.5 * eps
            eplus = self.model((species, coords @ x), cell @ x, pbc).energies
            x[i, j] = x[j, i] = -0.5 * eps
            eminus = self.model((species, coords @ x), cell @ x, pbc).energies
            virial[i, j] = virial[j, i] = (eplus - eminus) / (2 * eps)
        return virial * ase.units.Hartree

    def _dummy_tensor_req_grad(self) -> Tensor:
        return torch.empty(
            0, dtype=self._dtype, device=self._device, requires_grad=True
        )


def from_ase(
    atoms: ase.Atoms, device: Device = None, dtype: DType = None
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    species = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    coords = torch.tensor(
        atoms.get_positions(),
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    cell = torch.tensor(atoms.get_cell(complete=True).array, dtype=dtype, device=device)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    if not pbc.any():
        return species, coords, None, None
    return species, coords, cell, pbc


def _set_atom_positions_from_tensor(atoms: ase.Atoms, coords: Tensor) -> None:
    if coords.ndim == 3:
        coords = coords.squeeze(0)
    atoms.set_positions(coords.detach().cpu().numpy())
