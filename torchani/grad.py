r"""Wrapper functions that make use of `torch.autograd`

Computation of forces, hessians, and vibrational frequencies.
"""

import typing as tp

from torch import Tensor

from torchani.assembly import ANI
from torchani.potentials import Potential
from torchani.tuples import EnergiesForcesHessians, EnergiesForces
from torchani._grad import (
    forces,
    forces_for_training,
    hessians,
    forces_and_hessians,
    vibrational_analysis,
)

__all__ = [
    "energies_and_forces",
    "energies_forces_and_hessians",
    "forces",
    "forces_and_hessians",
    "forces_for_training",
    "hessians",
    "vibrational_analysis",
]

Model = tp.Union[Potential, ANI]


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
        energies = model(species, coordinates, cell=cell, pbc=pbc)
    else:
        energies = model((species, coordinates), cell=cell, pbc=pbc).energies
    _forces = forces(
        energies,
        coordinates,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    coordinates.requires_grad_(saved_requires_grad)
    return EnergiesForces(energies, _forces)
