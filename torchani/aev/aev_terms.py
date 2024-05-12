import math
from pathlib import Path

import torch
from torch import Tensor
from torch.jit import Final
import typing_extensions as tpx

from torchani.cutoffs import parse_cutoff_fn, CutoffArg

state_dicts_path = Path(__file__).parent.parent.joinpath("resources/state_dicts/")


class StandardRadial(torch.nn.Module):
    """Compute the radial sub-AEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    computes the terms. The sum in the equation is not computed.  The input
    tensor has shape (conformations, atoms, N), where ``N`` is the number of
    neighbor atoms within the cutoff radius and the output tensor should have
    shape (conformations, atoms, ``self.sublength``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    cutoff: Final[float]
    sublength: Final[int]
    EtaR: Tensor
    ShfR: Tensor

    def __init__(
        self, EtaR: Tensor, ShfR: Tensor, cutoff: float, cutoff_fn: CutoffArg = "cosine"
    ):
        super().__init__()
        # initialize the cutoff function
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        self.register_buffer("EtaR", EtaR.view(-1, 1))
        self.register_buffer("ShfR", ShfR.view(1, -1))
        self.sublength = self.EtaR.numel() * self.ShfR.numel()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        distances = distances.view(-1, 1, 1)
        fc = self.cutoff_fn(distances, self.cutoff)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        ret = 0.25 * torch.exp(-self.EtaR * (distances - self.ShfR) ** 2) * fc
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?) where ? depend on constants.
        # We then should flat the last 2 dimensions to view the sub-AEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)

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
        ShfR = torch.linspace(start, cutoff, int(num_shifts) + 1)[:-1].to(torch.float)
        EtaR = torch.tensor([eta], dtype=torch.float)
        return cls(EtaR, ShfR, cutoff, cutoff_fn)

    @classmethod
    def style_1x(
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
    def style_2x(
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

    @classmethod
    def like_1x(cls) -> tpx.Self:
        m = cls.style_1x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "radial_1x_state_dict.pth"))
        return m

    @classmethod
    def like_1ccx(cls) -> tpx.Self:
        m = cls.style_1x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "radial_1ccx_state_dict.pth"))
        return m

    @classmethod
    def like_2x(cls) -> tpx.Self:
        m = cls.style_2x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "radial_2x_state_dict.pth"))
        return m


class StandardAngular(torch.nn.Module):
    """Compute the angular sub-AEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum is not computed.  The input tensor has shape
    (conformations, atoms, N), where N is the number of neighbor atom pairs
    within the cutoff radius and the output tensor should have shape
    (conformations, atoms, ``self.sublength``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """

    sublength: Final[int]
    cutoff: Final[float]
    EtaA: Tensor
    Zeta: Tensor
    ShfA: Tensor
    ShfZ: Tensor

    def __init__(
        self,
        EtaA: Tensor,
        Zeta: Tensor,
        ShfA: Tensor,
        ShfZ: Tensor,
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ):
        super().__init__()
        # initialize the cutoff function
        self.cutoff_fn = parse_cutoff_fn(cutoff_fn)

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        self.register_buffer("EtaA", EtaA.view(-1, 1, 1, 1))
        self.register_buffer("Zeta", Zeta.view(1, -1, 1, 1))
        self.register_buffer("ShfA", ShfA.view(1, 1, -1, 1))
        self.register_buffer("ShfZ", ShfZ.view(1, 1, 1, -1))
        self.sublength = (
            self.EtaA.numel()
            * self.Zeta.numel()
            * self.ShfA.numel()
            * self.ShfZ.numel()
        )
        self.cutoff = cutoff

    def forward(self, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1, 1, 1)
        distances12 = vectors12.norm(2, dim=-5)
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(
            distances12.prod(0), min=1e-10
        )
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)

        fcj12 = self.cutoff_fn(distances12, self.cutoff)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(-self.EtaA * (distances12.sum(0) / 2 - self.ShfA) ** 2)
        # Use `fcj12[0] * fcj12[1]` instead of `fcj12.prod(0)` to avoid the INFs/NaNs
        # problem for smooth cutoff function, for more detail please check issue:
        # https://github.com/roitberg-group/torchani_sandbox/issues/178
        ret = 2 * factor1 * factor2 * (fcj12[0] * fcj12[1])
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the sub-AEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)

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
        EtaA = torch.tensor([eta], dtype=torch.float)
        ShfA = torch.linspace(start, cutoff, int(num_shifts) + 1)[:-1].to(torch.float)
        Zeta = torch.tensor([zeta], dtype=torch.float)
        angle_start = math.pi / (2 * int(num_angle_sections))
        ShfZ = (torch.linspace(0, math.pi, int(num_angle_sections) + 1) + angle_start)[
            :-1
        ].to(torch.float)
        return cls(EtaA, Zeta, ShfA, ShfZ, cutoff, cutoff_fn)

    @classmethod
    def style_1x(
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
    def style_2x(
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

    @classmethod
    def like_1x(cls) -> tpx.Self:
        m = cls.style_1x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "angular_1x_state_dict.pth"))
        return m

    @classmethod
    def like_1ccx(cls) -> tpx.Self:
        m = cls.style_1x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "angular_1ccx_state_dict.pth"))
        return m

    @classmethod
    def like_2x(cls) -> tpx.Self:
        m = cls.style_2x(cutoff_fn="cosine")
        m.load_state_dict(torch.load(state_dicts_path / "angular_2x_state_dict.pth"))
        return m


def parse_angular_terms(angular_terms):
    if angular_terms == "ani1x":
        angular_terms = StandardAngular.like_1x()
    elif angular_terms == "ani2x":
        angular_terms = StandardAngular.like_2x()
    elif angular_terms == "ani1ccx":
        angular_terms = StandardAngular.like_1ccx()
    else:
        assert isinstance(
            angular_terms, torch.nn.Module
        ), "Custom angular terms should be a torch module"
        assert hasattr(
            angular_terms, "sublength"
        ), "Custom angular terms should have a sublength attribute"
        assert hasattr(
            angular_terms, "cutoff"
        ), "Custom angular terms should have a cutoff attribute"
    return angular_terms


def parse_radial_terms(radial_terms):
    if radial_terms == "ani1x":
        radial_terms = StandardRadial.like_1x()
    elif radial_terms == "ani2x":
        radial_terms = StandardRadial.like_2x()
    elif radial_terms == "ani1ccx":
        radial_terms = StandardRadial.like_1ccx()
    else:
        assert isinstance(
            radial_terms, torch.nn.Module
        ), "Custom radial terms should be a torch module"
        assert hasattr(
            radial_terms, "sublength"
        ), "Custom radial terms should have a sublength attribute"
        assert hasattr(
            radial_terms, "cutoff"
        ), "Custom radial terms should have a cutoff attribute"
    return radial_terms
