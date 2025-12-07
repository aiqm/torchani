import math
import typing as tp
import torch
from torch import Tensor
from torchani.neighbors import _parse_neighborlist, NeighborlistArg, Neighbors
from torchani.nn import SpeciesConverter
from torchani.aev import ANIRadial
from torchani.cutoffs import CutoffArg
from torchani.tuples import SpeciesEnergies


# Activation used in schnet
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._shift = torch.log(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        # Equivalent to equation in SchNet article
        return torch.nn.functional.softplus(x) + self._shift


class CFConv(torch.nn.Module):
    def __init__(
        self,
        # Unclear if schnet uses 300 with cutoff 30.0 or 301 with cutoff 30.1
        num_shifts: int = 300,
        embed_dim: int = 64,
        eta: float = 10.0,
        cutoff: float = 30.0,
        cutoff_fn: CutoffArg = "dummy",
    ) -> None:
        super().__init__()
        self._radial = ANIRadial.cover_linearly(
            start=0.0,
            cutoff=cutoff,
            num_shifts=num_shifts,
            eta=eta,
            cutoff_fn=cutoff_fn,
        )
        self._linear0 = torch.nn.Linear(num_shifts, embed_dim)
        self._ssp = ShiftedSoftplus()
        self._linear1 = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: Tensor, neighbors: Neighbors, expansion: tp.Optional[Tensor] = None
    ) -> Tensor:
        num_molecs, num_atoms, num_feats = x.shape
        if expansion is None:
            expansion = self.expand(neighbors.distances)  # shape (P, 300)
        # Out has shape (P, 64)
        filt = self._ssp(self._linear1(self._ssp(self._linear0(expansion))))
        # Now I have to get all neighbors and multiply all with the input features
        indices = neighbors.indices
        x = x.view(-1, num_feats)
        out_features = x.new_zeros(x.shape)
        out_features.index_add_(0, indices[1], x[indices[1]] * filt)
        out_features.index_add_(0, indices[0], x[indices[0]] * filt)
        return out_features.view(num_molecs, num_atoms, num_feats)

    def expand(self, x: Tensor) -> Tensor:
        return self._radial(x) / 0.25  # shape (P, 300)


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_shifts: int = 300,
        eta: float = 10.0,
        cutoff: float = 30.0,
        cutoff_fn: CutoffArg = "dummy",
    ) -> None:
        super().__init__()
        self._pre_linear = torch.nn.Linear(embed_dim, embed_dim)
        self._cfconv = CFConv(num_shifts, embed_dim, eta, cutoff, cutoff_fn)
        self._post_linear = torch.nn.Linear(embed_dim, embed_dim)
        self._ssp = ShiftedSoftplus()
        self._final_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: Tensor, neighbors: Neighbors, expansion: tp.Optional[Tensor] = None
    ) -> Tensor:
        out = self._ssp(
            self._post_linear(self._cfconv(self._pre_linear(x), neighbors, expansion))
        )
        return x + self._final_linear(out)

    def expand(self, x: Tensor) -> Tensor:
        return self._cfconv.expand(x)


class SchNet(torch.nn.Module):
    def __init__(
        self,
        symbols: tp.Sequence[str],
        embed_dim: int = 64,
        num_shifts: int = 300,
        interaction_num: int = 3,
        eta: float = 10.0,
        neighborlist: NeighborlistArg = "all_pairs",
        cutoff: float = 30.0,
        cutoff_fn: CutoffArg = "dummy",
    ) -> None:
        # NOTE: The original SchNet has *no cutoff and no cutoff functions*
        super().__init__()
        # Atoms with index 0 are padding atoms
        self._symbols = symbols
        if cutoff_fn == "dummy":
            self._cutoff = math.inf
        else:
            self._cutoff = cutoff
        self._embedding = torch.nn.Embedding(
            len(symbols) + 1, embedding_dim=embed_dim, padding_idx=0
        )
        self._neighborlist = _parse_neighborlist(neighborlist)
        self._species_converter = SpeciesConverter(symbols)
        if interaction_num <= 0:
            raise ValueError("Number of interaction blocks must be >= 0")
        self._interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(embed_dim, num_shifts, eta, cutoff, cutoff_fn)
                for _ in range(interaction_num)
            ]
        )
        self._linear = torch.nn.Linear(embed_dim, embed_dim // 2)
        self._ssp = ShiftedSoftplus()
        self._final_linear = torch.nn.Linear(embed_dim // 2, 1)
        # NOTE: The original SchNet has *no energy shifter*

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        if ensemble_values:
            raise ValueError("Ensemble values not supported for SchNet")
        if charge != 0:
            raise ValueError("Non zero charge not supported for SchNet")
        if _molecule_idxs is not None:
            raise ValueError("_molecule_idxs not supported for SchNet")

        species, coords = species_coordinates
        elem_idxs = self._species_converter(species)
        neighbors = self._neighborlist(self._cutoff, elem_idxs, coords, cell, pbc)
        features = self._embedding(elem_idxs + 1)  # shape (N A)
        # Pre-calculate the expansion for efficiency
        expansion = self._interaction_blocks[0].expand(  # type: ignore[operator]
            neighbors.distances
        )
        for block in self._interaction_blocks:
            features = block(features, neighbors, expansion)
        out = self._ssp(self._linear(features))
        atomic_energies = self._final_linear(out).squeeze(-1)
        if atomic:
            return SpeciesEnergies(species, atomic_energies)
        return SpeciesEnergies(species, atomic_energies.sum(-1))
