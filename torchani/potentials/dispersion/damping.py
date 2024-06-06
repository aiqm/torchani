import typing as tp

import torch
from torch import Tensor
from torch.jit import Final

from torchani.constants import ATOMIC_NUMBER
from torchani.units import ANGSTROM_TO_BOHR
from torchani.potentials.dispersion import constants


# D3M modifies parameters AND damp function for zero-damp and only
# parameters for BJ damp cutoff radii are used for damp functions
class Damp(torch.nn.Module):
    r"""Damp function interface

    Damp functions are like cutoff functions, but modulate potentials close to
    zero.

    For modulating potentials of different "order" (e.g. 1 / r ** 6 => order 6),
    different parameters may be needed.
    """

    _order: Final[int]
    atomic_numbers: Tensor

    def __init__(
        self,
        *args,
        symbols: tp.Sequence[str] = ('H', 'C', 'N', 'O'),
        order: int = 6,
        **kwargs,
    ):
        super().__init__()
        self._order = order
        self.atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols],
            dtype=torch.long
        )

    @classmethod
    def from_functional(cls, functional: str = "wB97X", **kwargs) -> "Damp":
        raise NotImplementedError()

    def forward(self, species12: Tensor, distances: Tensor) -> Tensor:
        raise NotImplementedError()


class BJDamp(Damp):
    r"""Implementation of Becke-Johnson style damping

    For this damping style, the cutoff radii are by default calculated directly
    from the order 8 and order 6 coeffs, via the square root of the effective
    charges. Note that the cutoff radii is a matrix of T x T where T are the
    possible atom types and that these cutoff radii are in AU (Bohr)
    """
    cutoff_radii: Tensor
    _a1: Final[float]
    _a2: Final[float]

    def __init__(
        self,
        a1: float,
        a2: float,
        *args,
        cutoff_radii: tp.Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sqrt_q = constants.get_sqrt_empirical_charge()

        znumbers = self.atomic_numbers
        if cutoff_radii is None:
            _cutoff_radii = torch.sqrt(
                3 * torch.outer(sqrt_q, sqrt_q)
            )[:, znumbers][znumbers, :]
        else:
            _cutoff_radii = cutoff_radii

        # Cutoff radii is a matrix of T x T where T are the supported elements.
        assert _cutoff_radii.shape == (len(znumbers), len(znumbers))

        self.register_buffer('cutoff_radii', _cutoff_radii)
        self._a1 = a1
        self._a2 = a2

    @classmethod
    def from_functional(
        cls,
        functional: str = "wB97X",
        modified_damp: bool = False,
        **kwargs,
    ) -> "BJDamp":
        if modified_damp:
            raise ValueError("Modified damp is not yet implemented")
        d = constants.get_functional_constants()[functional.lower()]
        return cls(a1=d["a1"], a2=d["a2"], cutoff_radii=None, **kwargs)

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        damp_term = (self._a1 * cutoff_radii + self._a2).pow(self._order)
        return distances.pow(self._order) + damp_term


class ZeroDamp(Damp):
    r"""Zero-style damping

    Sometimes this is useful, but it may have some artifacts.
    TODO: This damping is untested
    """

    cutoff_radii: Tensor
    _sr: Final[float]
    _beta: Final[float]
    _alpha: Final[int]

    def __init__(
        self,
        alpha: int,
        sr: float,
        *args,
        beta: float = 0.0,
        cutoff_radii: tp.Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        znumbers = self.atomic_numbers
        if cutoff_radii is None:
            # These cutoff radii are in Angstrom, so we convert to Bohr.
            _cutoff_radii = ANGSTROM_TO_BOHR * constants.get_cutoff_radii()
            _cutoff_radii = _cutoff_radii[:, znumbers][znumbers, :]
        else:
            _cutoff_radii = cutoff_radii

        # Cutoff radii is a matrix of T x T where T are the supported elements.
        assert _cutoff_radii.shape == (len(znumbers), len(znumbers))

        self._sr = sr
        self._beta = beta
        self._alpha = alpha
        self.register_buffer('cutoff_radii', _cutoff_radii)

    @classmethod
    def from_functional(
        cls,
        functional: str = "wB97X",
        order: int = 6,
        modified_damp: bool = False,
        **kwargs
    ) -> "ZeroDamp":
        d = constants.get_functional_constants()[functional.lower()]
        if modified_damp:
            raise ValueError("Modified damp is not yet implemented")

        if order == 6:
            sr = d["sr6"]
            alpha = 14

        if order == 8:
            sr = d["sr8"]
            alpha = 16

        return cls(sr=sr, alpha=alpha, beta=0.0, cutoff_radii=None, **kwargs)

    def forward(
        self,
        species12: Tensor,
        distances: Tensor,
    ) -> Tensor:
        cutoff_radii = self.cutoff_radii[species12[0], species12[1]]
        inner_term = distances / (self._srr * cutoff_radii) + cutoff_radii * self._beta
        return distances.pow(self._order) * (1 + (6 * inner_term).pow(-self._alpha))


def _parse_damp_fn_cls(kind: str) -> tp.Type[Damp]:
    if kind == "bj":
        return BJDamp
    elif kind == "zero":
        return ZeroDamp
    raise ValueError("Incorrect damp function class")
