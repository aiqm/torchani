r"""
Holds base class for loss terms and some simple loss terms used to train
ANI-style models
"""

import typing as tp
from dataclasses import dataclass, asdict
from enum import Enum

import torch
from torch import Tensor
from typer import Abort
from rich.console import Console

from torchani.annotations import PyScalar


console = Console()


class Penalty(Enum):
    SQUARE = "square"
    ABS = "abs"


@dataclass
class LossTerm:
    label: str
    is_vec3: bool  # Vec3 quantities are scaled by 3 (after sum)
    is_atomic: bool  # Atomic quantities are normalized by 1/N (after sum)
    is_extensive: bool  # Extensive quantities (E) are normalized by 1/N (after sum)
    # targ_label_only is label in the dtaset, if blank assumed the same as 'label'
    targ_label_only: str = ""
    scale_by_sqrt_atoms: bool = False
    # Transform targ and pred to a per-atom quantity (1/N) *before* the loss
    make_per_atom: bool = False
    factor: float = 1.0
    grad_of_label: str = ""
    grad_wrt_targ_label: str = "coordinates"
    negative_grad: bool = False
    penalty: Penalty = Penalty.SQUARE

    @property
    def targ_label(self) -> str:
        return self.targ_label_only or self.label

    def as_dict(self) -> tp.Dict[str, PyScalar]:
        d = asdict(self)
        d["penalty"] = d["penalty"].value
        return d

    @property
    def needs_num_atoms_scaling(self) -> bool:
        return self.is_extensive or self.is_atomic or self.scale_by_sqrt_atoms


def Forces(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="forces",
        is_vec3=True,
        is_atomic=True,
        is_extensive=False,
        grad_of_label="energies",
        factor=factor,
        negative_grad=True,
    )


def UnnormalizedForces(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="forces",
        is_vec3=True,
        # Avoid dividing by 1/N, this is done in the MACE article, together
        # with EnergiesPerAtom
        is_atomic=False,
        is_extensive=False,
        grad_of_label="energies",
        factor=factor,
        negative_grad=True,
    )


def EnergiesPerAtom(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_vec3=False,
        is_atomic=False,
        # Since the resulting quantity is per-atom, it is not extensive in this case
        is_extensive=False,
        make_per_atom=True,
        factor=factor,
    )


def Energies(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_vec3=False,
        is_atomic=False,
        is_extensive=True,
        factor=factor,
    )


def UnnormalizedEnergies(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        # Avoid dividing by 1/N, this is done in the PhysNet and SchNet articles
        is_vec3=False,
        is_atomic=False,
        is_extensive=False,
        factor=factor,
    )


def EnergiesSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_vec3=False,
        is_atomic=False,
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
    )


def TotalCharge(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="total_charge",
        is_vec3=False,
        is_atomic=False,
        is_extensive=True,
        factor=factor,
    )


def EnergiesXC(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_vec3=False,
        is_atomic=False,
        is_extensive=True,
        factor=factor,
    )


def EnergiesXCSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_vec3=False,
        is_atomic=False,
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
    )


def EnergiesXCPerAtom(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_vec3=False,
        is_atomic=False,
        is_extensive=False,
        make_per_atom=True,
        factor=factor,
    )


def Dipoles(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="dipoles",
        is_vec3=True,
        is_atomic=False,
        is_extensive=False,
        factor=factor,
    )


def AtomicVolumes(
    factor: float = 1.0, targ_label: str = "atomic_volumes_mbis"
) -> LossTerm:
    # TODO: change to volumes
    return LossTerm(
        label="atomic_charges",
        is_vec3=False,
        is_atomic=True,
        is_extensive=False,
        targ_label_only=targ_label,
        factor=factor,
    )


def AtomicCharges(
    factor: float = 1.0, targ_label: str = "atomic_charges_mbis"
) -> LossTerm:
    return LossTerm(
        label="atomic_charges",
        is_vec3=False,
        is_atomic=True,
        is_extensive=False,
        targ_label_only=targ_label,
        factor=factor,
    )


class MultiTaskLoss(torch.nn.Module):
    r"""
    Represents a loss with multiple objectives (potentially scalar or vector
    valued)
    """

    def is_enabled(self, label: str) -> bool:
        r"""
        True if a specific label is being used in the loss
        """
        return any(term.label == label for term in self.terms)

    def term(self, label: str) -> LossTerm:
        for t in self.terms:
            if t.label == label:
                return t
        raise ValueError("Label not found")

    @property
    def grad_terms(self) -> tp.Iterator[LossTerm]:
        for term in self.terms:
            if term.grad_of_label:
                yield term

    def __init__(
        self,
        terms: tp.Sequence[LossTerm],
        uncertainty_weighted: bool = False,
    ) -> None:
        super().__init__()
        self.terms = tuple(terms)
        if len(self.terms) != len(set(term.label for term in self.terms)):
            raise ValueError("Loss terms must have unique labels")

        if uncertainty_weighted:
            raise NotImplementedError("Uncertainty Weighted loss not implemented yet")

    def forward(
        self,
        pred: tp.Dict[str, Tensor],
        targ: tp.Dict[str, Tensor],
    ) -> tp.Dict[str, Tensor]:
        r"""
        Calculate a dictionary of losses for some given predicted and target properties

        As an example, if the enabled loss terms are 'energies' and 'forces',
        then the output loss dictionary will be
        {'loss': <full-loss>, 'energies': <energy-loss>, 'forces': <force-loss>}.

        'species' must be one of the target properties
        """
        if "species" not in targ:
            raise ValueError("'species' must be one of the target properties")
        losses: tp.Dict[str, Tensor] = {}

        losses["loss"] = torch.tensor(
            0.0, dtype=torch.float, device=targ["species"].device
        )
        num_atoms = (targ["species"] >= 0).sum(dim=1, dtype=torch.float)
        for term in self.terms:
            if term.label not in pred:
                console.print(
                    f"Loss has {term.label} but model doesn't predict it", style="red"
                )
                raise Abort()

            # For forces: error.shape = (N, A, 3)
            # For atomic_charges: error.shape = (N, A)
            # For dipoles: error.shape = (N, 3)
            # For energies: error.shape = (N,)
            difference = pred[term.label] - targ[term.targ_label]

            # Transform difference into a per-atom quantity if needed
            if term.make_per_atom:
                difference /= num_atoms

            if term.penalty is Penalty.SQUARE:
                error = difference.pow(2)
            elif term.penalty is Penalty.ABS:
                error = difference.abs()

            # Sum over everything except batch size
            error = error.view(error.size(0), -1).sum(-1)

            # Calculate scaling after summation
            if term.scale_by_sqrt_atoms:
                error *= num_atoms.sqrt()
            if term.is_extensive:
                error /= num_atoms
            if term.is_atomic:
                error /= num_atoms
            if term.is_vec3:
                error = error / 3

            # Mean over the batch size
            mean_error = error.mean()
            losses[term.label] = mean_error
            losses["loss"] += mean_error * term.factor
        return losses


def build_loss_terms_and_factors(
    energies: float,
    forces: float,
    dipoles: float,
    atomic_charges: float,
    atomic_volumes: float,
    total_charge: float,
    normalize_energy_by_sqrt_atoms: bool = False,
    use_per_atom_energy: bool = False,
    use_unnormalized_forces: bool = False,
    use_unnormalized_energy: bool = False,
    use_xc_energies: bool = False,
) -> dict[str, float]:
    # Validate options

    if (
        use_per_atom_energy + normalize_energy_by_sqrt_atoms + use_unnormalized_energy
        > 1
    ) and energies == 0.0:
        raise ValueError("Energy options are mutually exclusive")
    if use_unnormalized_forces and (forces == 0.0):
        raise ValueError(
            "Forces factor must be provided if using force scaling options"
        )

    # Set terms and factors required to build the loss function
    terms_and_factors: tp.Dict[str, float] = {}
    if energies > 0.0:
        label = "Energies"
        if use_xc_energies:
            label = f"{label}XC"
        if normalize_energy_by_sqrt_atoms:
            label = f"{label}SqrtAtoms"
        if use_per_atom_energy:
            label = f"{label}PerAtom"
        if use_unnormalized_energy:
            label = f"Unnormalized{label}"
        terms_and_factors[label] = energies
    if forces > 0.0:
        label = "Forces"
        if use_unnormalized_forces:
            label = f"Unnormalized{label}"
        terms_and_factors[label] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_volumes > 0.0:
        terms_and_factors["AtomicVolumes"] = atomic_volumes
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge
    return terms_and_factors
