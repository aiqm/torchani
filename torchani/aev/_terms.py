import typing as tp
import math

import torch
from torch import Tensor
import typing_extensions as tpx

from torchani.cutoffs import _parse_cutoff_fn, CutoffArg
from torchani.utils import linspace, _validate_user_kwargs


class _Term(torch.nn.Module):
    cutoff: float
    num_feats: int

    def __init__(
        self,
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ) -> None:
        super().__init__()
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff
        self.num_feats = 0


class BaseAngular(_Term):
    r"""Base class for 3-body expansions

    Client classes should only rely on ``module(tri_distances, tri_vectors)``,
    ``compute_*`` functions are meant only to be overriden by users.
    """

    def forward(self, tri_distances: Tensor, tri_vectors: Tensor) -> Tensor:
        r""":meta private:"""
        # Wraps computation of terms with cutoff function
        assert tri_vectors.shape == (2, tri_distances.shape[1], 3)
        assert tri_distances.shape == (2, tri_distances.shape[1])

        tri_factor = self.cutoff_fn(tri_distances, self.cutoff)

        tri_vectors = tri_vectors.view(2, -1, 3, 1)
        tri_distances = tri_distances.view(2, -1, 1)
        cos_angles = (tri_vectors[0] * tri_vectors[1]).sum(1) / torch.clamp(
            tri_distances[0] * tri_distances[1], min=1e-10
        )
        # cos_angles = torch.clamp(cos_angles, min=-1.0, max=1.0)
        _angular = self.compute_cos_angles(cos_angles)
        _radial = self.compute_radial(tri_distances[0], tri_distances[1])
        terms = (_radial.unsqueeze(2) * _angular.unsqueeze(1)).view(-1, self.num_feats)
        # Use `fcj12[0] * fcj12[1]` instead of `fcj12.prod(0)` to avoid the INFs/NaNs
        # problem for smooth cutoff function, for more detail please check issue:
        # https://github.com/roitberg-group/torchani_sandbox/issues/178
        # shape (T, num-feats)
        return terms * (tri_factor[0] * tri_factor[1]).view(-1, 1)

    def compute_radial(self, distances_ji: Tensor, distances_jk) -> Tensor:
        r"""Computes the radial part of the 3-body terms

        This function *must* be overriden by subclasses

        Note:
            Don't call this method directly, instead call, ``module(triple_distances,
            triple_vectors)``.
        Args:
            distances_ji: Distances from the center atom, 'j', to a side atom, 'i'.
            distances_jk: Distances from the center atom, 'j', to a side atom, 'k'.
        Returns:
            A float `torch.Tensor` of shape ``(triples, feats)``. By design
            this function does *not* sum over atoms.
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def compute_cos_angles(self, cos_angles: Tensor) -> Tensor:
        r"""Computes the angular part of the 3-body terms

        This function *must* be overriden by subclasses

        Note:
            Don't call this method directly, instead call, ``module(triples.distances,
            triples.diff_vectors)``.
        Args:
            cos_angles: Cosine of the angle for a triple ijk
        Returns:
            A float `torch.Tensor` of shape ``(triples, feats)``. By design
            this function does *not* sum over atoms.
        """
        raise NotImplementedError("Must be implemented by subclasses")


# TODO: Change name to BaseRadial
class BaseRadial(_Term):
    r"""Base class for 2-body expansions

    Client classes should only rely on ``module(distances)``,
    ``compute`` is meant only to be overriden by users.
    """

    def forward(self, distances: Tensor) -> Tensor:
        r""":meta private:"""
        assert distances.shape == (distances.shape[0],)
        # Wraps computation of terms with cutoff function
        factor = self.cutoff_fn(distances, self.cutoff).view(-1, 1)
        return self.compute(distances.view(-1, 1)) * factor

    def compute(self, distances: Tensor) -> Tensor:
        r"""Compute the radial terms. Output shape is: ``(pairs, self.num_feats)``

        Subclasses must implement this method

        Note:
            Don't call this method directly, instead call,
            ``module(neighbors.distances)``.
        """
        raise NotImplementedError("Must be implemented by subclasses")


class ANIRadial(BaseRadial):
    r"""Compute the radial sub-AEV terms given a sequence of atom pair distances

    This correspond to equation (3) in the `ANI paper`_. This function just
    computes the terms. The sum in the equation is not computed.  The input
    tensor has shape (conformations, atoms, N), where ``N`` is the number of
    neighbor atoms within the cutoff radius and the output tensor should have
    shape (conformations, atoms, ``self.num_feats``)

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
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.num_feats = len(shifts)

    def extra_repr(self) -> str:
        r""":meta private:"""
        _shifts = [f"{s:.4f}" for s in self.shifts]
        parts = [
            r"#  " f"num_feats={self.num_feats}",
            r"#  " f"num_shifts={len(self.shifts)}",
            f"eta={self.eta.item():.4f},",
            f"shifts=[{', '.join(_shifts)}],",
            f"cutoff={self.cutoff:.4f},",
        ]
        return " \n".join(parts)

    def compute(self, distances: Tensor) -> Tensor:
        r"""Computes the ANI terms associated with a group of pairs

        Note:
            Don't call this method directly, instead call the module,
            ``module(distances)``.
        Args:
            distances: |distances|
        Returns:
            A float `torch.Tensor` of shape ``(pairs, shifts)``. By design
            this function does *not* sum over atoms.
        """
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        return 0.25 * torch.exp(-self.eta * (distances - self.shifts.view(1, -1)) ** 2)

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 5.2,
        eta: float = 19.7,
        num_shifts: int = 16,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        r"""Builds angular terms by linearly dividing space radially up to a cutoff

        ``num_shifts`` are created, starting from ``start`` until ``cutoff``, excluding
        it. This is the way angular and radial shifts were originally created in the
        `ANI paper`_.

        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
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


class ANIAngular(BaseAngular):
    """Compute the angular sub-AEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum is not computed.  The input tensor has shape
    (conformations, atoms, N), where N is the number of neighbor atom pairs
    within the cutoff radius and the output tensor should have shape
    (conformations, atoms, ``self.num_feats``)

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
                state_dict["".join((prefix, "sections"))] = state_dict.pop(k).view(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(
        self,
        eta: float,
        zeta: float,
        shifts: tp.Sequence[float],
        sections: tp.Sequence[float],
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        dtype = torch.float
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("zeta", torch.tensor([zeta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.register_buffer("sections", torch.tensor(sections, dtype=dtype))
        self.num_feats = len(shifts) * len(sections)

    def extra_repr(self) -> str:
        r""":meta private:"""
        _shifts = [f"{s:.4f}" for s in self.shifts]
        _sections = [f"{s:.4f}" for s in self.sections]
        parts = [
            r"#  " f"num_feats={self.num_feats}",
            r"#  " f"num_shifts={len(self.shifts)}",
            r"#  " f"num_sections={len(self.sections)}",
            f"eta={self.eta.item():.4f},",
            f"zeta={self.zeta.item():.4f},",
            f"shifts=[{', '.join(_shifts)}],",
            f"sections=[{', '.join(_sections)}],",
            f"cutoff={self.cutoff:.4f},",
        ]
        return " \n".join(parts)

    def compute_radial(self, distances_ji: Tensor, distances_jk) -> Tensor:
        r"""Computes the radial part of the ANI 3-body terms

        Note:
            Don't call this method directly, instead call, ``module(triple_distances,
            triple_vectors)``.
        Args:
            distances_ji: Distances from the center atom, 'j', to a side atom, 'i'.
            distances_jk: Distances from the center atom, 'j', to a side atom, 'k'.
        Returns:
            A float `torch.Tensor` of shape ``(triples, feats)``. By design
            this function does *not* sum over atoms.
        """
        mean_dists = (distances_ji + distances_jk) / 2
        return torch.exp(-self.eta * (mean_dists - self.shifts.view(1, -1)) ** 2)

    def compute_cos_angles(self, cos_angles: Tensor) -> Tensor:
        r"""Computes the angular part of the ANI 3-body terms

        Note:
            Don't call this method directly, instead call,
            ``module(triples.distances, triples.diff_vectors)``.
        Args:
            cos_angles: Cosine of the angle for a triple ijk
        Returns:
            A float `torch.Tensor` of shape ``(triples, feats)``. By design
            this function does *not* sum over atoms.
        """
        # The 0.95 is here for consistency with the original implementation,
        # where it was used to prevent NaNs due to cos_angles outside the [-1, 1] range.
        angles = torch.acos(0.95 * cos_angles)
        angle_deviations = angles - self.sections.view(1, -1)
        return 2 * ((1 + torch.cos(angle_deviations)) / 2) ** self.zeta

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 3.5,
        eta: float = 12.5,
        zeta: float = 14.1,
        num_shifts: int = 8,
        num_sections: int = 4,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        r"""Builds angular terms by dividing angular and radial coords, up to a cutoff

        The divisions are equally spaced "num_shifts" are created, starting from "start"
        until "cutoff", excluding it. "num_sections" does a similar thing for the
        angles. This is the way angular and radial shifts were originally created in
        ANI.
        """
        shifts = linspace(start, cutoff, num_shifts)
        angle_start = math.pi / num_sections / 2
        sections = linspace(angle_start, math.pi + angle_start, num_sections)
        return cls(eta, zeta, shifts, sections, cutoff, cutoff_fn)

    @classmethod
    def like_1x(
        cls,
        start: float = 0.9,
        cutoff: float = 3.5,
        eta: float = 8.0,
        zeta: float = 32.0,
        num_shifts: int = 4,
        num_sections: int = 8,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            zeta=zeta,
            num_shifts=num_shifts,
            num_sections=num_sections,
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
        num_sections: int = 4,
        cutoff_fn: CutoffArg = "cosine",
    ) -> tpx.Self:
        return cls.cover_linearly(
            start=start,
            cutoff=cutoff,
            eta=eta,
            zeta=zeta,
            num_shifts=num_shifts,
            num_sections=num_sections,
            cutoff_fn=cutoff_fn,
        )


# Classes meant for extension by users


class Angular(BaseAngular):

    angles_tensors: tp.List[str] = []
    radial_tensors: tp.List[str] = []

    def __init__(
        self,
        /,
        cutoff: float,
        trainable: tp.Union[str, tp.Sequence[str]] = (),
        cutoff_fn: CutoffArg = "cosine",
        **kwargs,
    ) -> None:
        super().__init__(cutoff, cutoff_fn)
        angles_feats = 1
        radial_feats = 1
        if isinstance(trainable, str):
            trainable = [trainable]

        _validate_user_kwargs(
            self.__class__.__name__,
            {
                "radial_tensors": self.radial_tensors,
                "angles_tensors": self.angles_tensors,
            },
            kwargs,
            trainable,
        )

        for k, v in kwargs.items():
            tensor = torch.tensor(v).view(1, -1)
            if k in trainable:
                self.register_parameter(k, torch.nn.Parameter(tensor))
            else:
                self.register_buffer(k, tensor)

            if k in self.angles_tensors:
                angles_feats = max(angles_feats, tensor.shape[1])
            else:
                radial_feats = max(radial_feats, tensor.shape[1])
        self.num_feats = radial_feats * angles_feats


class Radial(BaseRadial):

    tensors: tp.List[str] = []

    def __init__(
        self,
        /,
        cutoff: float,
        trainable: tp.Union[str, tp.Sequence[str]] = (),
        cutoff_fn: CutoffArg = "cosine",
        **kwargs,
    ) -> None:
        super().__init__(cutoff, cutoff_fn)
        self.num_feats = 1
        if isinstance(trainable, str):
            trainable = [trainable]

        _validate_user_kwargs(
            self.__class__.__name__,
            {
                "tensors": self.tensors,
            },
            kwargs,
            trainable,
        )
        for k, v in kwargs.items():
            tensor = torch.tensor(v).view(1, -1)
            if k in trainable:
                self.register_parameter(k, torch.nn.Parameter(tensor))
            else:
                self.register_buffer(k, tensor)
            self.num_feats = max(self.num_feats, tensor.shape[1])


_Models = tp.Literal["ani1x", "ani2x", "ani1ccx"]
AngularArg = tp.Union[_Models, BaseAngular]
RadialArg = tp.Union[_Models, BaseRadial]


def _parse_angular_term(angular_term: AngularArg) -> BaseAngular:
    if angular_term in ["ani1x", "ani1ccx"]:
        angular_term = ANIAngular.like_1x()
    elif angular_term == "ani2x":
        angular_term = ANIAngular.like_2x()
    elif not isinstance(angular_term, BaseAngular):
        raise ValueError(f"Unsupported angular term: {angular_term}")
    return tp.cast(BaseAngular, angular_term)


def _parse_radial_term(radial_term: RadialArg) -> BaseRadial:
    if radial_term in ["ani1x", "ani1ccx"]:
        radial_term = ANIRadial.like_1x()
    elif radial_term == "ani2x":
        radial_term = ANIRadial.like_2x()
    elif not isinstance(radial_term, BaseRadial):
        raise ValueError(f"Unsupported radial term: {radial_term}")
    return tp.cast(BaseRadial, radial_term)
