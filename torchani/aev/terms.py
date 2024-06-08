import typing as tp
import math

import torch
from torch import Tensor
import typing_extensions as tpx

from torchani.cutoffs import parse_cutoff_fn, CutoffArg
from torchani.utils import linspace


class _Term(torch.nn.Module):
    cutoff: float
    sublength: int

    def __init__(
        self,
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ) -> None:
        super().__init__()
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff
        self.sublength = 0


class AngularTerm(_Term):
    def forward(self, vectors: Tensor, distances: Tensor) -> Tensor:
        raise NotImplementedError("Must be implemented by subclasses")


class RadialTerm(_Term):
    def forward(self, distances: Tensor) -> Tensor:
        raise NotImplementedError("Must be implemented by subclasses")


class StandardRadial(RadialTerm):
    """Compute the radial sub-AEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    computes the terms. The sum in the equation is not computed.  The input
    tensor has shape (conformations, atoms, N), where ``N`` is the number of
    neighbor atoms within the cutoff radius and the output tensor should have
    shape (conformations, atoms, ``self.sublength``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            if suffix == "EtaR":
                value = state_dict.pop(k).view(-1)
                if value.numel() > 1:
                    raise RuntimeError("Only single 'eta' supported in standard terms")
                state_dict["".join((prefix, "eta"))] = value
            if suffix == "ShfR":
                state_dict["".join((prefix, "shifts"))] = state_dict.pop(k).view(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(
        self,
        eta: float,
        shifts: tp.Sequence[float],
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        dtype = torch.float
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.sublength = len(shifts)

    def extra_repr(self) -> str:
        _shifts = [f"{s:.4f}" for s in self.shifts]
        parts = [
            r"#  " f"sublength={self.sublength}",
            r"#  " f"num_shifts={len(self.shifts)}",
            f"eta={self.eta.item():.4f},",
            f"shifts=[{', '.join(_shifts)}],",
            f"cutoff={self.cutoff:.4f},",
        ]
        return " \n".join(parts)

    def forward(self, distances: Tensor) -> Tensor:
        distances = distances.view(-1, 1)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.eta * (distances - self.shifts.view(1, -1)) ** 2)
        ret *= self.cutoff_fn(distances, self.cutoff)
        return ret  # shape(P, radial_sublenght)

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 5.2,
        eta: float = 19.7,
        num_shifts: int = 16,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        r"""Builds angular terms by linearly subdividing space radially up to a cutoff

        "num_shifts" are created, starting from "start" until "cutoff",
        excluding it. This similar to the way angular and radial shifts were
        originally created for the ANI models
        """
        shifts = linspace(start, cutoff, num_shifts)
        return cls(eta, shifts, cutoff, cutoff_fn)

    @classmethod
    def like_1x(
        cls,
        start: float = 0.9,
        cutoff: float = 5.2,
        eta: float = 16.0,
        num_shifts: int = 16,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            num_shifts=num_shifts,
            cutoff_fn=cutoff_fn,
        )

    @classmethod
    def like_2x(
        cls,
        start: float = 0.8,
        cutoff: float = 5.1,
        eta: float = 19.7,
        num_shifts: int = 16,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            num_shifts=num_shifts,
            cutoff_fn=cutoff_fn,
        )


class StandardAngular(AngularTerm):
    """Compute the angular sub-AEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum is not computed.  The input tensor has shape
    (conformations, atoms, N), where N is the number of neighbor atom pairs
    within the cutoff radius and the output tensor should have shape
    (conformations, atoms, ``self.sublength``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    # Needed for bw compatibility
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        old_keys = list(state_dict.keys())
        for k in old_keys:
            suffix = k.split(prefix)[-1] if prefix else k
            if suffix == "EtaA":
                value = state_dict.pop(k).view(-1)
                if value.numel() > 1:
                    raise RuntimeError("Only single 'eta' supported in standard terms")
                state_dict["".join((prefix, "eta"))] = value
            if suffix == "Zeta":
                value = state_dict.pop(k).view(-1)
                if value.numel() > 1:
                    raise RuntimeError("Only single 'zeta' supported in standard terms")
                state_dict["".join((prefix, "zeta"))] = value
            if suffix == "ShfA":
                state_dict["".join((prefix, "shifts"))] = state_dict.pop(k).view(-1)
            if suffix == "ShfZ":
                state_dict["".join((prefix, "angle_sections"))] = state_dict.pop(
                    k
                ).view(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(
        self,
        eta: float,
        zeta: float,
        shifts: tp.Sequence[float],
        angle_sections: tp.Sequence[float],
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        dtype = torch.float
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("zeta", torch.tensor([zeta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.register_buffer(
            "angle_sections", torch.tensor(angle_sections, dtype=dtype)
        )
        self.sublength = len(shifts) * len(angle_sections)

    def extra_repr(self) -> str:
        _shifts = [f"{s:.4f}" for s in self.shifts]
        _angle_sections = [f"{s:.4f}" for s in self.angle_sections]
        parts = [
            r"#  " f"sublength={self.sublength}",
            r"#  " f"num_shifts={len(self.shifts)}",
            r"#  " f"num_angle_sections={len(self.angle_sections)}",
            f"eta={self.eta.item():.4f},",
            f"zeta={self.zeta.item():.4f},",
            f"shifts=[{', '.join(_shifts)}],",
            f"angle_sections=[{', '.join(_angle_sections)}],",
            f"cutoff={self.cutoff:.4f},",
        ]
        return " \n".join(parts)

    def forward(self, vectors12: Tensor, distances12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1)
        distances12 = distances12.view(2, -1, 1, 1)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10
        )
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)
        angle_deviations = angles - self.angle_sections.view(1, -1)
        factor1 = ((1 + torch.cos(angle_deviations)) / 2) ** self.zeta

        mean_distance_deviations = distances12.sum(0) / 2 - self.shifts.view(-1, 1)
        factor2 = torch.exp(-self.eta * mean_distance_deviations**2)

        fcj12 = self.cutoff_fn(distances12, self.cutoff)
        # Use `fcj12[0] * fcj12[1]` instead of `fcj12.prod(0)` to avoid the INFs/NaNs
        # problem for smooth cutoff function, for more detail please check issue:
        # https://github.com/roitberg-group/torchani_sandbox/issues/178
        # shape (T, shifts, sections)
        ret = 2 * factor1 * factor2 * (fcj12[0] * fcj12[1])
        # shape (T, sublength)
        return ret.view(-1, self.sublength)

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 3.5,
        eta: float = 12.5,
        zeta: float = 14.1,
        num_shifts: int = 8,
        num_angle_sections: int = 4,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        r"""Builds angular terms by linearly subdividing space in the angular
        dimension and in the radial one up to a cutoff

        "num_shifts" are created, starting from "start" until "cutoff",
        excluding it. "num_angle_sections" does a similar thing for the angles.
        This is the way angular and radial shifts were originally created in
        ANI.
        """
        shifts = linspace(start, cutoff, num_shifts)
        angle_start = math.pi / num_angle_sections / 2
        angle_sections = linspace(
            angle_start, math.pi + angle_start, num_angle_sections
        )
        return cls(eta, zeta, shifts, angle_sections, cutoff, cutoff_fn)

    @classmethod
    def like_1x(
        cls,
        start: float = 0.9,
        cutoff: float = 3.5,
        eta: float = 8.0,
        zeta: float = 32.0,
        num_shifts: int = 4,
        num_angle_sections: int = 8,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            zeta=zeta,
            num_shifts=num_shifts,
            num_angle_sections=num_angle_sections,
            cutoff_fn=cutoff_fn,
        )

    @classmethod
    def like_2x(
        cls,
        start: float = 0.8,
        cutoff: float = 3.5,
        eta: float = 12.5,
        zeta: float = 14.1,
        num_shifts: int = 8,
        num_angle_sections: int = 4,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            zeta=zeta,
            num_shifts=num_shifts,
            num_angle_sections=num_angle_sections,
            cutoff_fn=cutoff_fn,
        )


_Models = tp.Literal["ani1x", "ani2x", "ani1ccx"]
AngularTermArg = tp.Union[_Models, AngularTerm]
RadialTermArg = tp.Union[_Models, RadialTerm]


def parse_angular_term(angular_term: AngularTermArg) -> AngularTerm:
    if angular_term in ["ani1x", "ani1ccx"]:
        angular_term = StandardAngular.like_1x()
    elif angular_term == "ani2x":
        angular_term = StandardAngular.like_2x()
    elif not isinstance(angular_term, AngularTerm):
        raise ValueError(f"Unsupported angular term: {angular_term}")
    return tp.cast(AngularTerm, angular_term)


def parse_radial_term(radial_term: RadialTermArg) -> RadialTerm:
    if radial_term in ["ani1x", "ani1ccx"]:
        radial_term = StandardRadial.like_1x()
    elif radial_term == "ani2x":
        radial_term = StandardRadial.like_2x()
    elif not isinstance(radial_term, RadialTerm):
        raise ValueError(f"Unsupported radial term: {radial_term}")
    return tp.cast(RadialTerm, radial_term)
