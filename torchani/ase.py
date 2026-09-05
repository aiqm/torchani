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
from ase.calculators.calculator import (
    Calculator as AseCalculator,
    all_changes as _ALL_CHANGES,
)

from torchani.annotations import StressKind, Device, DType
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

    # NOTE: 'free energy' seems to mean smth slightly different in ASE context
    # it is what is get_potential_energy(force_consistent=True) returns
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self, model, overwrite: bool = False, stress_kind: StressKind = "scaling"
    ):
        super().__init__()
        if hasattr(model, "periodic_table_index") and not model.periodic_table_index:
            raise ValueError("ASE models must have periodic_table_index=True")
        if stress_kind not in ["fdotr", "scaling", "numerical"]:
            raise ValueError(f"Unsupported stress kind {stress_kind}")
        param = next(model.parameters())
        self.model = model
        self._device = param.device
        self._dtype = param.dtype
        self._overwrite = overwrite
        self._stress_kind = stress_kind

    # NOTE: The ASE default is _ALL_CHANGES ==
    # ["positions", "numbers", "cell", "pbc", "initial_charges", "initial_magmoms"]
    # NOTE: It is a bad idea to use mutables (e.g. lists) as defaults, but this is what
    # ASE does Since the functions don't mutate the arguments, it is harmless, (but
    # error-prone and a potential source of bugs, so be careful)
    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = _ALL_CHANGES,
    ):
        # NOTE: If atoms is passed, then the
        # superclass overwrites self.atoms with the passed atoms
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("Can't calculate if not attached to Atoms")

        species, crds, cell, pbc = from_ase(self.atoms, self._device, self._dtype)
        if "forces" in properties:
            crds.requires_grad_(True)

        strain = crds.new_zeros((3, 3))
        if "stress" in properties and self._stress_kind == "scaling":
            strain.requires_grad_(True)

        if self._overwrite and (cell is not None) and (pbc is not None):
            warnings.warn("'overwrite' set, info about crossing PBC *will be lost*")
            crds = map_to_central(crds, cell, pbc)
            self.atoms.set_positions(crds.squeeze(0).detach().cpu().numpy())

        # Forwards pass
        if "stress" in properties:
            if cell is None or pbc is None:
                raise ValueError("Can't require stress if not using PBC")

            if self._stress_kind == "scaling":
                eye = torch.eye(3, dtype=self._dtype, device=self._device)
                # Sym not strictly needed but done for numeric robustness
                defo_grad = eye + 0.5 * (strain + strain.T)
                crds = crds @ defo_grad
                cell = cell @ defo_grad

            _cutoff = self.model.cutoff
            neighbors = self.model.neighborlist(_cutoff, species, crds, cell, pbc)

            if self._stress_kind == "fdotr":
                diff_vec = neighbors.diff_vectors
                diff_vec.requires_grad_(True)
                neighbors = Neighbors(neighbors.indices, diff_vec.norm(2, -1), diff_vec)

            _idxs = self.model.species_converter(species)
            energy = self.model.compute_from_neighbors(_idxs, crds, neighbors).energies
        else:
            energy = self.model((species, crds), cell, pbc).energies
        energy = energy * ase.units.Hartree

        # Check if properties requires a backwards pass, if so run it
        if self._calc_needs_autograd(properties):
            # [crds, scaling|diff_vec]
            inputs = self._dummy_autograd_inputs(num=2)
            if "forces" in properties:
                inputs[0] = crds
            if "stress" in properties:
                if self._stress_kind == "fdotr":
                    inputs[1] = diff_vec
                elif self._stress_kind == "scaling":
                    inputs[1] = strain
            grads = torch.autograd.grad(energy.squeeze(), inputs, allow_unused=True)

        # Set all calculated properties in the "results" mapping
        if "energy" in properties:
            self._set_result("energy", energy)
        if "free_energy" in properties:
            self._set_result("free_energy", energy)
        if "forces" in properties:
            self._set_result("forces", -grads[0])
        if "stress" in properties:
            volume = torch.linalg.det(cell).abs()
            if self._stress_kind == "numerical":
                virial = self._numerical_virial(species, crds, cell, pbc)
            elif self._stress_kind == "scaling":
                virial = grads[1]
            elif self._stress_kind == "fdotr":
                virial = grads[1].transpose(0, 1) @ diff_vec  # grads[1] == dE/ddiff_vec
            self._set_result("stress", virial / volume)

    def _calc_needs_autograd(self, properties: list[str]) -> bool:
        return "forces" in properties or (
            "stress" in properties and self._stress_kind != "numerical"
        )

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
    def _numerical_virial(
        self,
        species: Tensor,
        crds: Tensor,
        cell: Tensor | None,
        pbc: Tensor | None,
        # A bit of a larger value for numerical stability
        eps: float = 1e-3,
    ) -> Tensor:
        if cell is None or pbc is None:
            raise ValueError("Cell is required for calculating numerical virial")
        virial = torch.zeros((3, 3), dtype=self._dtype, device=self._device)
        for i in range(3):
            # Diagonal terms
            x = torch.eye(3, dtype=self._dtype, device=self._device)
            x[i, i] = 1.0 + eps
            eplus = self.model((species, crds @ x), cell @ x, pbc).energies
            x[i, i] = 1.0 - eps
            eminus = self.model((species, crds @ x), cell @ x, pbc).energies
            virial[i, i] = (eplus - eminus) / (2 * eps)

            # Off diagonal terms
            x = torch.eye(3, dtype=self._dtype, device=self._device)  # Reset
            j = i - 2
            x[i, j] = x[j, i] = +0.5 * eps
            eplus = self.model((species, crds @ x), cell @ x, pbc).energies
            x[i, j] = x[j, i] = -0.5 * eps
            eminus = self.model((species, crds @ x), cell @ x, pbc).energies
            virial[i, j] = virial[j, i] = (eplus - eminus) / (2 * eps)
        return virial * ase.units.Hartree

    def _dummy_autograd_inputs(self, num: int) -> list[Tensor]:
        return [
            torch.empty(0, dtype=self._dtype, device=self._device, requires_grad=True)
            for _ in range(num)
        ]


def from_ase(
    atoms: ase.Atoms, device: Device = None, dtype: DType = None
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, None, None]:
    species = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    crds = torch.tensor(
        atoms.get_positions(),
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    cell = torch.tensor(atoms.get_cell(complete=True).array, dtype=dtype, device=device)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    if not pbc.any():
        return species, crds, None, None
    return species, crds, cell, pbc
