import math
from typing import Tuple, Optional, NamedTuple
import warnings
import importlib_metadata

import torch
from torch import Tensor

from ..utils import cumsum_from_zero
from ..compat import Final
# modular parts of AEVComputer
from .cutoffs import _parse_cutoff_fn, CutoffCosine, CutoffSmooth
from .aev_terms import _parse_angular_terms, _parse_radial_terms, StandardAngular, StandardRadial
from .neighbors import _parse_neighborlist


cuaev_is_installed = 'torchani.cuaev' in importlib_metadata.metadata(
    __package__.split('.')[0]).get_all('Provides')

if cuaev_is_installed:
    # We need to import torchani.cuaev to tell PyTorch to initialize torch.ops.cuaev
    from .. import cuaev  # type: ignore # noqa: F401
else:
    warnings.warn("cuaev not installed")


class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor


def jit_unused_if_no_cuaev(condition=cuaev_is_installed):
    def decorator(func):
        if not condition:
            return torch.jit.unused(func)
        return torch.jit.export(func)
    return decorator


class AEVComputer(torch.nn.Module):
    r"""The AEV computer that takes coordinates as input and outputs aevs.

    Arguments:
        Rcr (float): :math:`R_C` in equation (2) when used at equation (3)
            in the `ANI paper`_.
        Rca (float): :math:`R_C` in equation (2) when used at equation (4)
            in the `ANI paper`_.
        EtaR (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (3) in the `ANI paper`_.
        ShfR (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (3) in the `ANI paper`_.
        EtaA (:class:`torch.Tensor`): The 1D tensor of :math:`\eta` in
            equation (4) in the `ANI paper`_.
        Zeta (:class:`torch.Tensor`): The 1D tensor of :math:`\zeta` in
            equation (4) in the `ANI paper`_.
        ShfA (:class:`torch.Tensor`): The 1D tensor of :math:`R_s` in
            equation (4) in the `ANI paper`_.
        ShfZ (:class:`torch.Tensor`): The 1D tensor of :math:`\theta_s` in
            equation (4) in the `ANI paper`_.
        num_species (int): Number of supported atom types.
        use_cuda_extension (bool): Whether to use cuda extension for faster calculation (needs cuaev installed).

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    num_species: Final[int]
    num_species_pairs: Final[int]

    angular_length: Final[int]
    angular_sublength: Final[int]
    radial_length: Final[int]
    radial_sublength: Final[int]
    aev_length: Final[int]

    use_cuda_extension: Final[bool]
    triu_index: Tensor

    def __init__(self,
                Rcr: Optional[float] = None,
                Rca: Optional[float] = None,
                EtaR: Optional[Tensor] = None,
                ShfR: Optional[Tensor] = None,
                EtaA: Optional[Tensor] = None,
                Zeta: Optional[Tensor] = None,
                ShfA: Optional[Tensor] = None,
                ShfZ: Optional[Tensor] = None,
                num_species: Optional[int] = None,
                use_cuda_extension=False,
                cutoff_fn='cosine',
                neighborlist='full_pairwise',
                radial_terms='standard',
                angular_terms='standard'):

        # due to legacy reasons num_species is a kwarg, but it should always be
        # provided
        assert num_species is not None, "num_species should be provided to construct an AEVComputer"

        super().__init__()
        self.use_cuda_extension = use_cuda_extension
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2

        # currently only cosine, smooth and custom cutoffs are supported
        # only ANI-1 style angular terms or radial terms
        # and only full pairwise neighborlist
        # if a cutoff function is passed, it is used for both radial and
        # angular terms.
        cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.angular_terms = _parse_angular_terms(angular_terms, cutoff_fn, EtaA, Zeta, ShfA, ShfZ, Rca)
        self.radial_terms = _parse_radial_terms(radial_terms, cutoff_fn, EtaR, ShfR, Rcr)
        self.neighborlist = _parse_neighborlist(neighborlist, self.radial_terms.cutoff)
        self._validate_cutoffs_init()
        if isinstance(cutoff_fn, CutoffCosine):
            self.cutoff_fn_type = 'cosine'
        elif isinstance(cutoff_fn, CutoffSmooth):
            if cutoff_fn.order == 2 and cutoff_fn.eps == 1e-10:
                self.cutoff_fn_type = 'smooth'
            else:
                self.cutoff_fn_type = 'smooth_modified'
        else:
            self.cutoff_fn_type = 'others'

        self.register_buffer('triu_index',
                             self._calculate_triu_index(num_species).to(device=self.radial_terms.EtaR.device))

        # length variables are updated once radial and angular terms are initialized
        # The lengths of buffers can't be changed with load_state_dict so we can
        # cache all lengths in the model itself
        self.radial_sublength = self.radial_terms.sublength
        self.angular_sublength = self.angular_terms.sublength
        self.radial_length = self.radial_sublength * self.num_species
        self.angular_length = self.angular_sublength * self.num_species_pairs
        self.aev_length = self.radial_length + self.angular_length

        # cuda aev
        if self.use_cuda_extension:
            assert cuaev_is_installed, "AEV cuda extension is not installed"
            assert isinstance(self.angular_terms, StandardAngular), 'nonstandard aev terms not supported for cuaev'
            assert isinstance(self.radial_terms, StandardRadial), 'nonstandard aev terms not supported for cuaev'
        if cuaev_is_installed:
            self._register_cuaev_computer()

        # We defer true cuaev initialization to forward so that we ensure that
        # all tensors are in GPU once it is initialized.
        self.cuaev_is_initialized = False

    def _validate_cutoffs_init(self):
        # validate cutoffs and emit warnings for strange configurations
        if self.neighborlist.cutoff > self.radial_terms.cutoff:
            raise ValueError(f"""The neighborlist cutoff {self.neighborlist.cutoff}
                    is larger than the radial cutoff,
                    {self.radial_terms.cutoff}.  please fix this since
                    AEVComputer can't possibly reuse the neighborlist for other
                    interactions, so this configuration would not use the extra
                    atom pairs""")
        elif self.neighborlist.cutoff < self.radial_terms.cutoff:
            raise ValueError(f"""The neighborlist cutoff,
                             {self.neighborlist.cutoff} should be at least as
                             large as the radial cutoff, {self.radial_terms.cutoff}""")
        if self.angular_terms.cutoff > self.radial_terms.cutoff:
            raise ValueError(f"""Current implementation assumes angular cutoff
                             {self.angular_terms.cutoff} < radial cutoff
                             {self.radial_terms.cutoff}""")

    @jit_unused_if_no_cuaev()
    def _register_cuaev_computer(self):
        # cuaev_computer is created only when use_cuda_extension is True.
        # However jit needs to know cuaev_computer's Type even when
        # use_cuda_extension is False. **this is only a kind of "dummy"
        # initialization, it is always necessary to reinitialize in forward at
        # least once, since some tensors may be on CPU at this point**
        empty = torch.empty(0)
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(0.0, 0.0, empty, empty, empty, empty, empty, empty, 1, True)

    @jit_unused_if_no_cuaev()
    def _init_cuaev_computer(self):
        assert self.cutoff_fn_type != 'others', 'cuaev currently only supports cosine and smooth cutoff functions'
        assert self.cutoff_fn_type != 'smooth_modified', 'cuaev currently only supports standard parameters for smooth cutoff function'
        use_cos_cutoff = self.cutoff_fn_type == 'cosine'
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(self.radial_terms.cutoff,
                                                                self.angular_terms.cutoff,
                                                                self.radial_terms.EtaR.flatten(),
                                                                self.radial_terms.ShfR.flatten(),
                                                                self.angular_terms.EtaA.flatten(),
                                                                self.angular_terms.Zeta.flatten(),
                                                                self.angular_terms.ShfA.flatten(),
                                                                self.angular_terms.ShfZ.flatten(),
                                                                self.num_species,
                                                                use_cos_cutoff)

    @staticmethod
    def _calculate_triu_index(num_species: int) -> Tensor:
        # helper method for initialization
        species1, species2 = torch.triu_indices(num_species,
                                                num_species).unbind(0)
        pair_index = torch.arange(species1.shape[0], dtype=torch.long)
        ret = torch.zeros(num_species, num_species, dtype=torch.long)
        ret[species1, species2] = pair_index
        ret[species2, species1] = pair_index
        return ret

    @classmethod
    def cover_linearly(cls,
                       radial_cutoff: float,
                       angular_cutoff: float,
                       radial_eta: float,
                       angular_eta: float,
                       radial_dist_divisions: int,
                       angular_dist_divisions: int,
                       zeta: float,
                       angle_sections: int,
                       num_species: int,
                       angular_start: float = 0.9,
                       radial_start: float = 0.9, **kwargs):
        warnings.warn('cover_linearly is deprecated')
        Rcr = radial_cutoff
        Rca = angular_cutoff
        EtaR = torch.tensor([radial_eta], dtype=torch.float)
        EtaA = torch.tensor([angular_eta], dtype=torch.float)
        Zeta = torch.tensor([zeta], dtype=torch.float)
        ShfR = torch.linspace(radial_start, radial_cutoff,
                              radial_dist_divisions + 1)[:-1].to(torch.float)
        ShfA = torch.linspace(angular_start, angular_cutoff,
                              angular_dist_divisions + 1)[:-1].to(torch.float)
        angle_start = math.pi / (2 * angle_sections)
        ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1].to(torch.float)
        return cls(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, **kwargs)

    @classmethod
    def like_1x(cls, **kwargs):
        return cls(angular_terms='ani1x', radial_terms='ani1x', num_species=4, **kwargs)

    @classmethod
    def like_2x(cls, **kwargs):
        return cls(angular_terms='ani2x', radial_terms='ani2x', num_species=7, **kwargs)

    @classmethod
    def like_1ccx(cls, **kwargs):
        # just a synonym
        return cls.like_1x(**kwargs)

    def forward(self,
                input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`torchani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, 3) of the
                three vectors defining unit cell:

                .. code-block:: python

                    tensor([[x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3]])

                and pbc is boolean vector of size 3 storing if pbc is enabled
                for that direction.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length)``
        """
        species, coordinates = input_
        # check shapes for correctness
        assert species.dim() == 2
        assert coordinates.dim() == 3
        assert (species.shape == coordinates.shape[:2]) and (coordinates.shape[2] == 3)

        # validate cutoffs
        assert self.neighborlist.cutoff >= self.radial_terms.cutoff
        assert self.angular_terms.cutoff < self.radial_terms.cutoff

        if self.use_cuda_extension:
            if not self.cuaev_is_initialized:
                self._init_cuaev_computer()
                self.cuaev_is_initialized = True
            assert pbc is None or (not pbc.any()), "cuaev currently does not support PBC"
            aev = self._compute_cuaev(species, coordinates)
            return SpeciesAEV(species, aev)

        # WARNING: The coordinates that are input into the neighborlist are **not** assumed to be
        # mapped into the central cell for pbc calculations,
        # and **in general are not**
        atom_index12, _, diff_vector, distances = self.neighborlist(species, coordinates, cell, pbc)
        aev = self._compute_aev(species, atom_index12, diff_vector, distances)
        return SpeciesAEV(species, aev)

    @jit_unused_if_no_cuaev()
    def _compute_cuaev(self, species, coordinates):
        species_int = species.to(torch.int32)
        coordinates = coordinates.to(torch.float)
        aev = torch.ops.cuaev.run(coordinates, species_int, self.cuaev_computer)
        return aev

    def _compute_aev(self, species: Tensor,
            atom_index12: Tensor, diff_vector: Tensor, distances: Tensor) -> Tensor:

        species12 = species.flatten()[atom_index12]
        radial_aev = self._compute_radial_aev(species.shape[0], species.shape[1], species12,
                                              distances, atom_index12)

        # Rca is usually much smaller than Rcr, using neighbor list with
        # cutoff = Rcr is a waste of resources. Now we will get a smaller neighbor
        # list that only cares about atoms with distances <= Rca
        even_closer_indices = (distances <= self.angular_terms.cutoff).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        diff_vector = diff_vector.index_select(0, even_closer_indices)

        angular_aev = self._compute_angular_aev(species.shape[0], species.shape[1], species12,
                                                diff_vector, atom_index12)

        return torch.cat([radial_aev, angular_aev], dim=-1)

    def _compute_angular_aev(self, num_molecules: int, num_atoms: int, species12: Tensor, vec: Tensor,
                             atom_index12: Tensor) -> Tensor:

        central_atom_index, pair_index12, sign12 = self._triple_by_molecule(
            atom_index12)
        species12_small = species12[:, pair_index12]
        vec12 = vec.index_select(0, pair_index12.view(-1)).view(
            2, -1, 3) * sign12.unsqueeze(-1)
        species12_ = torch.where(sign12 == 1, species12_small[1],
                                 species12_small[0])

        angular_terms_ = self.angular_terms(vec12)
        angular_aev = angular_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species_pairs,
             self.angular_sublength))
        index = central_atom_index * self.num_species_pairs + self.triu_index[
            species12_[0], species12_[1]]
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(num_molecules, num_atoms,
                                          self.angular_length)
        return angular_aev

    def _compute_radial_aev(self, num_molecules: int, num_atoms: int, species12: Tensor, distances: Tensor,
                            atom_index12: Tensor) -> Tensor:

        radial_terms_ = self.radial_terms(distances)
        radial_aev = radial_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species,
             self.radial_sublength))
        index12 = atom_index12 * self.num_species + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_terms_)
        radial_aev.index_add_(0, index12[1], radial_terms_)
        radial_aev = radial_aev.reshape(num_molecules, num_atoms,
                                        self.radial_length)
        return radial_aev

    def _triple_by_molecule(
            self, atom_index12: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Input: indices for pairs of atoms that are close to each other.
        each pair only appear once, i.e. only one of the pairs (1, 2) and
        (2, 1) exists.

        Output: indices for all central atoms and it pairs of neighbors. For
        example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
        central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
        are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
        """
        # convert representation from pair to central-others
        ai1 = atom_index12.view(-1)
        sorted_ai1, rev_indices = ai1.sort()

        # sort and compute unique key
        uniqued_central_atom_index, counts = torch.unique_consecutive(
            sorted_ai1, return_inverse=False, return_counts=True)

        # compute central_atom_index
        pair_sizes = (counts * (counts - 1)).div(2, rounding_mode='floor')
        pair_indices = torch.repeat_interleave(pair_sizes)
        central_atom_index = uniqued_central_atom_index.index_select(
            0, pair_indices)

        # do local combinations within unique key, assuming sorted
        m = counts.max().item() if counts.numel() > 0 else 0
        n = pair_sizes.shape[0]
        intra_pair_indices = torch.tril_indices(
            m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
        mask = (torch.arange(intra_pair_indices.shape[2], device=ai1.device) < pair_sizes.unsqueeze(1)).flatten()
        sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
        sorted_local_index12 += cumsum_from_zero(counts).index_select(
            0, pair_indices)

        # unsort result from last part
        local_index12 = rev_indices[sorted_local_index12]

        # compute mapping between representation of central-other to pair
        n = atom_index12.shape[1]
        sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
        return central_atom_index, local_index12 % n, sign12
