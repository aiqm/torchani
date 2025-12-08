r"""
Holds base class for loss terms and some simple loss terms used to train
ANI-style models
"""

import typing as tp
from dataclasses import dataclass, asdict
from enum import Enum

import torch
from torch import Tensor

from torchani.annotations import PyScalar


class Penalty(Enum):
    SQUARE = "square"
    ABS = "abs"


@dataclass
class LossTerm:
    label: str = "pred"
    # targ_label_only is label in the dataset, if blank assumed the same as 'label'
    targ_label_only: str = ""
    is_extensive: bool = False
    scale_by_sqrt_atoms: bool = False
    is_vec3: bool = False
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


def Forces(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="forces",
        grad_of_label="energies",
        is_vec3=True,
        is_extensive=False,
        factor=factor,
        negative_grad=True,
    )


def Energies(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_extensive=True,
        factor=factor,
    )


def EnergiesSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
    )


def TotalCharge(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="total_charge",
        is_extensive=True,
        factor=factor,
    )


def EnergiesXC(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_extensive=True,
        factor=factor,
    )


def EnergiesXCSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
    )


def Dipoles(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="dipoles",
        is_extensive=False,
        is_vec3=True,
        factor=factor,
    )


def AtomicVolumes(
    factor: float = 1.0, targ_label: str = "atomic_volumes_mbis"
) -> LossTerm:
    # TODO: change to volumes
    return LossTerm(
        label="atomic_charges",
        targ_label_only=targ_label,
        is_extensive=True,
        factor=factor,
    )


def AtomicCharges(
    factor: float = 1.0, targ_label: str = "atomic_charges_mbis"
) -> LossTerm:
    return LossTerm(
        label="atomic_charges",
        targ_label_only=targ_label,
        is_extensive=True,
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
        for term in self.terms:
            if term.penalty is Penalty.SQUARE:
                error = (pred[term.label] - targ[term.targ_label]).pow(2)
            elif term.penalty is Penalty.ABS:
                #  For forces: error.shape = (N, A, 3)
                error = torch.abs(pred[term.label] - targ[term.targ_label])

            if term.scale_by_sqrt_atoms or term.is_extensive:
                num_atoms = (targ["species"] >= 0).sum(dim=1, dtype=torch.float)
                num_atoms = num_atoms.view((-1,) + (1,) * (error.ndim - 1))
                if term.scale_by_sqrt_atoms:
                    error *= num_atoms.sqrt()
                if term.is_extensive:
                    error /= num_atoms

            if term.is_vec3:
                error = error / 3

            # Sum over everything except batch size
            error = error.view(error.size(0), -1).sum(-1)
            mean_error = error.mean()
            losses[term.label] = mean_error
            losses["loss"] += mean_error * term.factor
        return losses
