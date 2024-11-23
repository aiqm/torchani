import math
import typing as tp

import torch
from torch import Tensor

from torchani.tuples import EnergiesScalars
from torchani.cutoffs import _parse_cutoff_fn, CutoffArg, CutoffDummy
from torchani.neighbors import Neighbors, adaptive_list, all_pairs
from torchani.utils import _validate_user_kwargs
from torchani.units import ANGSTROM_TO_BOHR
from torchani._core import _ChemModule


class Potential(_ChemModule):
    r"""Base class for all atomic potentials

    Potentials may be many-body (3-body, 4-body, ...) potentials or 2-body (pair)
    potentials. Subclasses must implement ``compute`` and may override ``__init__``.
    """

    ANGSTROM_TO_BOHR: float
    cutoff: float
    _enabled: bool

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        cutoff: float = math.inf,
    ):
        super().__init__(symbols)
        self.cutoff = cutoff
        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR
        self._enabled = True  # Currently meant for other classes to access only

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
        atomic_nums_input: bool = True,
    ) -> Tensor:
        r"""
        Outputs energy, as calculated by the potential

        Output shape depends on the value of ``atomic``, it is either
        ``(molecs, atoms)`` or ``(molecs,)``
        """
        if atomic_nums_input:
            elem_idxs = self._conv_tensor.to(species.device)[species]
        else:
            elem_idxs = species
        # Check inputs
        assert elem_idxs.dim() == 2
        assert coords.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)

        if coords.shape[0] == 1:
            neighbors = adaptive_list(self.cutoff, elem_idxs, coords, cell, pbc)
        else:
            neighbors = all_pairs(self.cutoff, elem_idxs, coords, cell, pbc)
        return self.compute_from_neighbors(
            elem_idxs, coords, neighbors, atomic, ensemble_values
        ).energies

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        r"""Compute the energies associated with the potential

        If the potential is an ensemble of multiple models, ensemble_values=True should
        return the individual values of the models, in the first dimension ``(submodels,
        ...)``. Otherwise it should *disregard* ``ensemble_values``.

        Must be implemented by subclasses
        """
        raise NotImplementedError("Must be implemented by subclasses")


class DummyPotential(Potential):
    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        if atomic:
            return EnergiesScalars(coords.new_zeros(elem_idxs.shape))
        return EnergiesScalars(coords.new_zeros(elem_idxs.shape[0]))


class BasePairPotential(Potential):
    r"""General base class for all pairwise potentials

    Subclasses must implement pair_energies, and override init using this template:

    .. code-block:: python

        def __init__(
            symbols,
            ..., # User args go here
            cutoff: float=math.inf,
            cutoff_fn="smooth",
            ..., # User kwargs go here
        )
            super().__init__(symbols, cutoff, cutoff_fn)
            ... # User code goes here
    """

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff)
        if cutoff != math.inf:
            self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        else:
            self.cutoff_fn = CutoffDummy()

    @staticmethod
    def clamp(distances: Tensor) -> Tensor:
        return distances.clamp(min=1e-7)

    def pair_energies(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
    ) -> Tensor:
        r"""Return energy of all pairs of neighbors

        Returns a Tensor of energies, of shape ('pairs',) where 'pairs' is
        the number of neighbor pairs.
        """
        raise NotImplementedError("Must be overriden by subclasses")

    # Modulate pair_energies by wrapping with the cutoff fn envelope,
    # and scale ghost pair energies by 0.5
    def _pair_energies_wrapper(
        self,
        elem_idxs: Tensor,
        neighbors: Neighbors,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> Tensor:
        # Input validation
        assert elem_idxs.ndim == 2, "species should be 2 dimensional"
        assert neighbors.distances.ndim == 1, "distances should be 1 dimensional"
        assert neighbors.indices.ndim == 2, "atom_index12 should be 2 dimensional"
        assert neighbors.distances.shape[0] == neighbors.indices.shape[1]

        pair_energies = self.pair_energies(elem_idxs, neighbors)
        pair_energies *= self.cutoff_fn(neighbors.distances, self.cutoff)

        if ghost_flags is not None:
            if not ghost_flags.numel() == elem_idxs.numel():
                raise ValueError(
                    "ghost_flags and species should have the same number of elements"
                )
            ghost12 = ghost_flags.view(-1)[neighbors.indices]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    # Compute_form_neighbors should not be modified by subclasses of BasePairPotential
    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        # NOTE: Currently having ensembles of pair potentials is not supported, so
        # ensemble_values is disregarded
        # NOTE: Currently charge is not passed to the pair_potentials
        pair_energies = self._pair_energies_wrapper(elem_idxs, neighbors, ghost_flags)
        molecs_num, atoms_num = elem_idxs.shape
        if atomic:
            energies = neighbors.distances.new_zeros(molecs_num * atoms_num)
            energies.index_add_(0, neighbors.indices[0], pair_energies / 2)
            energies.index_add_(0, neighbors.indices[1], pair_energies / 2)
            energies = energies.view(molecs_num, atoms_num)
        else:
            energies = neighbors.distances.new_zeros(molecs_num)
            molecs_idxs = torch.div(
                neighbors.indices[0], elem_idxs.shape[1], rounding_mode="floor"
            )
            energies.index_add_(0, molecs_idxs, pair_energies)
        return EnergiesScalars(energies)

    @staticmethod
    def symm(x: Tensor) -> Tensor:
        r"""Takes an NxN tensor with 0 lower triangle and returns it symmetrized"""
        assert x.ndim == 2, "Arg to symm must be an N x N tensor"
        assert x.shape[0] == x.shape[1], "Arg to symm must be an N x N tensor"
        assert (x.tril(-1) == 0).all(), "Arg to symm must have 0 low triangle part"
        return x + x.triu(1).T

    def to_pair_values(self, x: Tensor, elem_idxs: Tensor) -> Tensor:
        r"""Returns values of elem pairs from a NxN tensor with 0 low triangle"""
        return self.symm(x)[elem_idxs[0], elem_idxs[1]]  # shape(num-elem, num-elem)


class PairPotential(BasePairPotential):
    r"""User friendly, simple class for pairwise potentials

    Subclasses must implement ``pair_energies`` and, if they use
    any paramters or buffers, specify three list of strings:

    - ``'tensors'``: Vectors (all with the same len) or scalars
    - ``'elem_tensors'``: With shape ``(num-sym,)``
    - ``'pair_elem_tensors'``: With shape ``(num-sym * (num-sym + 1) / 2,)``

    Usage is better understood by an example:

    .. code-block:: python

        from torchani.potentials import PairPotential


        class Square(PairPotential):
            tensors = ['bias']  # Vectors (all with the same len) or scalars
            pair_elem_tensors = ["k", "eq"]  # shape (num-sym * (num-sym + 1) / 2)

            def pair_energies(self, elem_idxs, neighbors):
                elem_pairs = elem_idxs.view(-1)[neighbors.indices]
                eq = self.to_pair_values(self.eq, elem_pairs)
                k = self.to_pair_values(self.k, elem_pairs)
                return self.bias + k / 2 * (neighbors.distances - eq) ** 2


        #  Order for the pair elem tensors is HH, HC, HO, CC, CO, ...
        #  Values for demonstration purpose only
        k = (1.,) * (3 * (3 + 1) // 2)
        eq = (1.5,) * (3 * (3 + 1) // 2)
        pot = Square(symbols=("H", "C", "O"), k=k, eq=eq, bias=0.1)

        # Or if the constants are trainable:
        pot = Square(symbols=("H", "C", "O"), k=k, bias=0.1, eq=eq, trainable="k")
    """

    tensors: tp.List[str] = []
    elem_tensors: tp.List[str] = []
    pair_elem_tensors: tp.List[str] = []

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        trainable: tp.Union[str, tp.Sequence[str]] = (),
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
        **kwargs,
    ) -> None:
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        if isinstance(trainable, str):
            trainable = [trainable]

        _validate_user_kwargs(
            self.__class__.__name__,
            {
                "tensors": self.tensors,
                "elem_tensors": self.elem_tensors,
                "pair_elem_tensors": self.pair_elem_tensors,
            },
            kwargs,
            trainable,
        )
        for k in self.elem_tensors:
            self._validate_elem_seq(k, kwargs.get(k))

        for k in self.pair_elem_tensors:
            self._validate_elem_seq(k, kwargs.get(k), pair=True)

        for k, v in kwargs.items():
            tensor = torch.tensor(v)
            if k in self.pair_elem_tensors:
                shape = (len(symbols),) * 2
                upper = tensor.new_zeros(shape)
                upper[torch.triu_indices(*shape).unbind()] = tensor
                tensor = upper
            if k in trainable:
                self.register_parameter(k, torch.nn.Parameter(tensor))
            else:
                self.register_buffer(k, tensor)
