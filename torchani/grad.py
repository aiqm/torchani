r"""Wrapper functions that make use of `torch.autograd`

Computation of forces, hessians, and vibrational frequencies.
"""

import math
import typing as tp

from torch import Tensor
import torch

from torchani.arch import ANI, ANIq
from torchani.potentials import Potential
from torchani.tuples import (
    EnergiesForcesHessians,
    EnergiesForces,
    VibAnalysis,
    ForcesHessians,
)
from torchani.units import mhessian2fconst, sqrt_mhessian2invcm, sqrt_mhessian2milliev

__all__ = [
    "single_point",
    "vibrational_analysis",
    "forces_for_training",
    # Convenience fn
    "energies_and_forces",
    "forces",
    "grads",
    "hessians",
    "forces_and_hessians",
    "energies_forces_and_hessians",
    "calc_forces",  # alias
    "calc_grads",  # alias
    "calc_hessians",  # alias
    "calc_forces_and_hessians",  # alias
]

Model = tp.Union[Potential, ANI, ANIq]


def forces(
    energies: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
) -> Tensor:
    if not coordinates.requires_grad:
        raise ValueError(
            "'coordinates' passed to `torchani.grad.forces` must require grad"
        )
    if not coordinates.is_leaf:
        raise ValueError(
            "'coordinates' passed to `torchani.grad` functions must be a 'leaf' Tensor"
            "(i.e. must not have been modified prior to being used as an input)."
        )
    grads = torch.autograd.grad(
        [energies.sum()],
        [coordinates],
        retain_graph=retain_graph,
        create_graph=create_graph,
    )[0]
    assert grads is not None  # JIT
    return -grads


# Alias for convenience, just flips - sign
def grads(
    scalars: Tensor,
    coords: Tensor,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
) -> Tensor:
    return -forces(scalars, coords, retain_graph, create_graph)


calc_forces = forces
calc_grads = grads


# Convenience function, for training create_graph=True and retain_graph=True
def forces_for_training(energies: Tensor, coordinates: Tensor) -> Tensor:
    return forces(energies, coordinates, retain_graph=True, create_graph=True)


def forces_and_hessians(
    energies: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
) -> ForcesHessians:
    _forces = forces(
        energies,
        coordinates,
        retain_graph=True,
        create_graph=True,
    )
    _hessians = hessians(
        _forces,
        coordinates,
        retain_graph=retain_graph,
    )
    return ForcesHessians(_forces, _hessians)


calc_forces_and_hessians = forces_and_hessians


def hessians(
    forces: Tensor,
    coordinates: Tensor,
    retain_graph: tp.Optional[bool] = None,
) -> Tensor:
    if not coordinates.requires_grad:
        raise ValueError(
            "'coordinates' passed to `torchani.grad.hessians` must require grad"
        )
    if not coordinates.is_leaf:
        raise ValueError(
            "'coordinates' passed to `torchani.grad` functions must be a 'leaf' Tensor"
            "(i.e. must not have been modified prior to being used as an input)."
        )
    num_molecules, num_atoms, num_dim = forces.shape
    num_components = num_atoms * num_dim
    flat_forces = forces.view(num_molecules, num_components)
    # 3A Tensors, each of shape (C,)
    flat_forces_tuple = flat_forces.unbind(dim=1)
    result_list = []
    _retain_graph: tp.Optional[bool] = True
    for j, component in enumerate(flat_forces_tuple):
        # For the last component we retain the graph only if instructed to
        if j == (num_components - 1):
            _retain_graph = retain_graph
        _grads = torch.autograd.grad(
            [component.sum()],
            [coordinates],
            retain_graph=_retain_graph,
        )[0]
        assert _grads is not None  # JIT
        _grads = _grads.view(num_molecules, 1, num_components)  # shape (C, 1, 3A)
        result_list.append(_grads)
    _hessians = -torch.cat(result_list, dim=1)  # shape (C, 3A, 3A)
    return _hessians


calc_hessians = hessians


ModeKind = tp.Literal["mdu", "mdn", "mwn"]
UnitKind = tp.Literal["cm^-1", "meV"]


def vibrational_analysis(
    masses: Tensor,
    hessian: Tensor,
    mode_kind: ModeKind = "mdu",
    unit: UnitKind = "cm^-1",
):
    """Computing the vibrational wavenumbers from hessian.

    Note that normal modes in many popular software packages such as
    Gaussian and ORCA are output as mass deweighted normalized (MDN).
    Normal modes in ASE are output as mass deweighted unnormalized (MDU).
    Some packages such as Psi4 let ychoose different normalizations.
    Force constants and reduced masses are calculated as in Gaussian.

    mode_kind should be one of:
    - MWN (mass weighted normalized)
    - MDU (mass deweighted unnormalized)
    - MDN (mass deweighted normalized)

    MDU modes are not orthogonal, and not normalized,
    MDN modes are not orthogonal, and normalized.
    MWN modes are orthonormal, but they correspond
    to mass weighted cartesian coordinates (x' = sqrt(m)x).

    Imaginary frequencies are output as negative numbers.
    Very small negative or positive frequencies may correspond to
    translational, and rotational modes.
    """
    if unit == "meV":
        converter = sqrt_mhessian2milliev
    elif unit == "cm^-1":
        converter = sqrt_mhessian2invcm
    else:
        raise ValueError("Only meV and cm^-1 are supported right now")

    assert (
        hessian.shape[0] == 1
    ), "Currently only supporting computing one molecule a time"
    # Solving the eigenvalue problem: Hq = w^2 * T q
    # where H is the Hessian matrix, q is the normal coordinates,
    # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
    # We solve this eigenvalue problem through Lowdin diagnolization:
    # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
    # Letting q' = T^(1/2) q, we then have
    # T^(-1/2) H T^(-1/2) q' = w^2 * q'
    inv_sqrt_mass = (1 / masses.sqrt()).repeat_interleave(3, dim=1)  # shape (C, 3 * A)
    mass_scaled_hessian = (
        hessian * inv_sqrt_mass.unsqueeze(1) * inv_sqrt_mass.unsqueeze(2)
    )
    if mass_scaled_hessian.shape[0] != 1:
        raise ValueError("The input should contain only one molecule")
    mass_scaled_hessian = mass_scaled_hessian.squeeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(mass_scaled_hessian)
    signs = torch.sign(eigenvalues)

    # Note that the normal modes are the COLUMNS of the eigenvectors matrix
    mw_normalized = eigenvectors.t()
    md_unnormalized = mw_normalized * inv_sqrt_mass
    norm_factors = 1 / torch.linalg.norm(md_unnormalized, dim=1)  # units are sqrt(AMU)
    rmasses = norm_factors**2  # units are AMU
    # The conversion factor for Ha/(AMU*A^2) to mDyne/(A*AMU) is about 4.3597482
    fconstants = mhessian2fconst(eigenvalues) * rmasses  # units are mDyne/A

    if mode_kind.lower() in ["mdn", "mass-deweighted-normalized"]:
        modes = (md_unnormalized * norm_factors.unsqueeze(1)).reshape(
            eigenvalues.numel(), -1, 3
        )
    elif mode_kind.lower() in ["mdu", "mass-deweighted-unnormalized"]:
        modes = (md_unnormalized).reshape(eigenvalues.numel(), -1, 3)
    elif mode_kind.lower() in ["mwn", "mass-weighted-normalized"]:
        modes = (mw_normalized).reshape(eigenvalues.numel(), -1, 3)
    else:
        raise ValueError(f"Incorrect mode kind {mode_kind}")

    # Converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 or meV
    angular_frequencies = eigenvalues.abs().sqrt()
    frequencies = angular_frequencies / (2 * math.pi)
    frequencies = frequencies * signs
    frequencies = converter(frequencies)
    return VibAnalysis(
        frequencies,
        modes,
        fconstants,
        rmasses,
    )


def energies_forces_and_hessians(
    model: Model,
    species: Tensor,
    coordinates: Tensor,
    retain_graph: bool = False,
) -> EnergiesForcesHessians:
    saved_requires_grad = coordinates.requires_grad
    coordinates.requires_grad_(True)
    energies, forces = energies_and_forces(
        model,
        species,
        coordinates,
        retain_graph=True,
        create_graph=True,
    )
    _hessians = hessians(
        forces,
        coordinates,
        retain_graph=retain_graph,
    )
    coordinates.requires_grad_(saved_requires_grad)
    return EnergiesForcesHessians(energies, forces, _hessians)


def energies_and_forces(
    model: Model,
    species: Tensor,
    coordinates: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    retain_graph: tp.Optional[bool] = None,
    create_graph: bool = False,
) -> EnergiesForces:
    saved_requires_grad = coordinates.requires_grad
    coordinates.requires_grad_(True)
    if not coordinates.is_leaf:
        raise ValueError(
            "'coordinates' passed to `torchani.grad` functions must be a 'leaf' Tensor"
            "(i.e. must not have been modified prior to being used as an input)."
        )
    if isinstance(model, Potential):
        energies = model(species, coordinates, cell, pbc)
    else:
        energies = model((species, coordinates), cell, pbc).energies
    _forces = forces(
        energies,
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    coordinates.requires_grad_(saved_requires_grad)
    return EnergiesForces(energies, _forces)


def single_point(
    model: tp.Union[ANI, ANIq],
    species: Tensor,
    coordinates: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
    charge: int = 0,
    forces: bool = False,
    hessians: bool = False,
    atomic_energies: bool = False,
    atomic_charges: bool = False,
    atomic_charges_grad: bool = False,
    ensemble_values: bool = False,
    keep_vars: bool = False,
) -> tp.Dict[str, Tensor]:
    r"""Calculate properties for a batch of molecules

    This is the main entrypoint of ANI-style models

    Args:
        species: |atomic_nums|
        coordinates: |coords|
        cell: |cell|
        pbc: |pbc|
        charge: The total charge of the molecules. Only
            the scalar 0 is currently supported.
        forces: Calculate the associated forces. Shape ``(molecules, atoms, 3)``
        hessians: Calculate the hessians. Shape is
            ``(molecules, atoms * 3, atoms * 3)``
        atomic_energies: Perform atomic decoposition of the energies
        atomic_charges: Only for models that support it, output atomic charges.
            Shape ``(molecules, atoms)``
        atomic_charges_grad: Only for models that support it, output atomic charge
            gradients. Shape ``(molecules, atoms, 3)``.
        ensemble_values: Differentiate values of different models of the ensemble
            Also output ensemble standard deviation and qbc factors
        keep_vars: The output scalars are detached from the graph unless
            ``keep_vars=True``.
    Returns:
        Result of the single point calculation. Dictionary that maps strings to
        various result tensors.
    """
    saved_requires_grad = coordinates.requires_grad
    if forces or hessians or atomic_charges_grad:
        coordinates.requires_grad_(True)
    result = model(
        species_coordinates=(species, coordinates),
        cell=cell,
        pbc=pbc,
        charge=charge,
        atomic=atomic_energies,
        ensemble_values=ensemble_values,
    )
    energies = result.energies
    out: tp.Dict[str, Tensor] = {}
    if atomic_charges:
        if not hasattr(result, "atomic_charges"):
            raise ValueError("Model doesn't support atomic charges")
        out["atomic_charges"] = result.atomic_charges
        if atomic_charges_grad:
            retain = forces or hessians
            out["atomic_charges_grad"] = calc_grads(
                result.atomic_charges, coordinates, retain_graph=retain
            )

    if ensemble_values:
        if atomic_energies:
            out["atomic_energies"] = energies.mean(dim=0)
            _values = energies.sum(dim=-1)
        else:
            _values = energies
        out["energies"] = _values.mean(dim=0)

        if _values.shape[0] == 1:
            out["ensemble_std"] = _values.new_zeros(energies.shape)
        else:
            out["ensemble_std"] = _values.std(dim=0, unbiased=True)
        out["ensemble_values"] = _values

        if _values.shape[0] == 1:
            qbc_factors = _values.new_zeros(_values.shape).squeeze(0)
        else:
            # std is taken across ensemble members
            qbc_factors = _values.std(0, unbiased=True)
        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        assert qbc_factors.shape == out["energies"].shape
        out["qbcs"] = qbc_factors
    else:
        if atomic_energies:
            out["energies"] = energies.sum(dim=-1)
            out["atomic_energies"] = energies
        else:
            out["energies"] = energies
    if hessians:
        _forces, _hessians = calc_forces_and_hessians(out["energies"], coordinates)
        out["forces"], out["hessians"] = _forces, _hessians
        if forces:
            out["forces"] = _forces
    elif forces:
        out["forces"] = calc_forces(out["energies"], coordinates)
    coordinates.requires_grad_(saved_requires_grad)
    if not keep_vars:
        out = {k: v.detach() for k, v in out.items()}
    return out
