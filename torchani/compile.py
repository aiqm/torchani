r"""Ahead-of-time compilation of models into AOTInductor packages

.. warning::

    This module is experimental. Its API and behavior may change without warning.

The compiled package holds one model per entrypoint, device and precision, named
``"<entrypoint>/<device>/<precision>"`` (e.g. ``"single_point_pbc/cuda/f32"``). The
models can be loaded without Python, from C++, using
``torch::inductor::AOTIModelPackageLoader(package_path, model_name)``, or from Python,
using ``torch._inductor.package.load_package(package_path, model_name)``. The
entrypoints and their signatures are:

- ``"single_point"``:
  ``(species, coords, net_charge) -> (energies, forces, atomic_charges)``
- ``"single_point_pbc"``:
  ``(species, coords, cell, pbc, net_charge) -> (energies, forces, atomic_charges)``
- ``"external_neighbors"``:
  ``(species, coords, neighbor_idxs, shifts, net_charge)
  -> (energies, forces, atomic_charges)``
- ``"custom_mlmm"``:
  ``(species, coords, mm_coords, mm_charges, net_charge, spin_mult, mm_cutoff,
  pairs_12, pairs_13, pairs_14) -> (energies, forces, mm_forces, atomic_charges)``
- ``"multipole_mlmm"``:
  ``(species, coords, mm_coords, mm_charges, mm_dipoles, mm_quads, net_charge,
  spin_mult, mm_cutoff, pairs_12, pairs_13, pairs_14)
  -> (energies, forces, mm_forces, atomic_charges)``

with input shapes ``species: (1, A)``, ``coords: (1, A, 3)``, ``cell: (3, 3)``,
``pbc: (3,)``, ``net_charge: (1,)``, ``spin_mult: (1,)``, ``neighbor_idxs: (2, P)``,
``shifts: (P, 3)``, ``mm_coords: (1, M, 3)``, ``mm_charges: (1, M)``,
``mm_dipoles: (1, M, 3)``, ``mm_quads: (1, M, 3, 3)``, ``mm_cutoff: (1,)``,
``pairs_12: (2, P12)``, ``pairs_13: (2, P13)`` and ``pairs_14: (2, P14)``, and
output shapes ``(1,)``, ``(1, A, 3)``, ``(1, M, 3)`` (``mm_forces``) and ``(1, A)``.
If the model doesn't predict atomic charges, they are an empty tensor of shape
``(1, 0)``. The ``"custom_mlmm"`` entrypoint requires a model that exports a
``compute_mlmm`` method, which accepts ``spin_mult``, ``mm_cutoff``, ``pairs_12``,
``pairs_13`` and ``pairs_14`` keyword arguments and returns either the energies or a
tuple ``(energies, atomic_charges)``. Models may also output a dict with the
``"energies"`` key and, optionally, ``"atomic_charges"``.

The ``"multipole_mlmm"`` entrypoint is the same, but it requires a
``compute_multipole_mlmm`` method, which also takes ``mm_dipoles`` and ``mm_quads``
after ``mm_charges``.

All inputs must be contiguous, and for ``"external_neighbors"`` the ``coords`` must be
mapped to the central cell. The CUDA models use cuAEV if the model supports it, and ANI
CPU models use pyaev. If a model uses cuAEV or ``FastCellList``, the corresponding
extension must be loaded by the process before running the package.
"""

import copy
import struct
import typing as tp
from pathlib import Path

import torch
from torch import Tensor
from torch.export import Dim, ExportedProgram
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.package import package_aoti

from torchani.constants import ATOMIC_NUMBER
from torchani.neighbors import (
    AllPairs,
    AdaptiveList,
    CellList,
    FastAdaptiveList,
    FastCellList,
)
from torchani.tuples import SpeciesEnergiesAtomicCharges, SpeciesEnergiesScalars

Device = tp.Literal["cpu", "cuda"]
Precision = tp.Literal["f32", "f64"]
Config = tp.Literal[
    "single_point",
    "single_point_pbc",
    "external_neighbors",
    "custom_mlmm",
    "multipole_mlmm",
]
_DTYPES = {"f32": torch.float32, "f64": torch.float64}


class _SinglePoint(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self, species: Tensor, coords: Tensor, net_charge: Tensor
    ) -> tuple[Tensor, Tensor]:
        result = self._model((species, coords), charge=net_charge)
        if isinstance(result, dict):
            empty_charges = coords.new_empty((1, 0))
            return result["energies"], result.get("atomic_charges", empty_charges)
        if isinstance(result, SpeciesEnergiesAtomicCharges):
            return result.energies, result.atomic_charges
        if isinstance(result, SpeciesEnergiesScalars):
            empty_charges = coords.new_empty((1, 0))
            return result.energies, result.scalars.get("atomic_charges", empty_charges)
        return result.energies, coords.new_empty((1, 0))


class _SinglePointPBC(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        cell: Tensor,
        pbc: Tensor,
        net_charge: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = self._model((species, coords), cell=cell, pbc=pbc, charge=net_charge)
        if isinstance(result, dict):
            empty_charges = coords.new_empty((1, 0))
            return result["energies"], result.get("atomic_charges", empty_charges)
        if isinstance(result, SpeciesEnergiesAtomicCharges):
            return result.energies, result.atomic_charges
        if isinstance(result, SpeciesEnergiesScalars):
            empty_charges = coords.new_empty((1, 0))
            return result.energies, result.scalars.get("atomic_charges", empty_charges)
        return result.energies, coords.new_empty((1, 0))


class _ExternalNeighbors(torch.nn.Module):
    def __init__(self, model: tp.Any) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        neighbor_idxs: Tensor,
        shifts: Tensor,
        net_charge: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = self._model.compute_from_external_neighbors(
            species, coords, neighbor_idxs, shifts, charge=net_charge
        )
        empty_charges = coords.new_empty((1, 0))
        return result["energies"], result.get("atomic_charges", empty_charges)


def _mlmm_energies_atomic_charges(
    result: tp.Any, coords: Tensor
) -> tuple[Tensor, Tensor]:
    if isinstance(result, dict):
        empty_charges = coords.new_empty((1, 0))
        return result["energies"], result.get("atomic_charges", empty_charges)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, coords.new_empty((1, 0))


class _CustomMLMM(torch.nn.Module):
    # The model is any module with a 'compute_mlmm' method
    def __init__(self, model: tp.Any) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        mm_coords: Tensor,
        mm_charges: Tensor,
        net_charge: Tensor,
        spin_mult: Tensor,
        mm_cutoff: Tensor,
        pairs_12: Tensor,
        pairs_13: Tensor,
        pairs_14: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = self._model.compute_mlmm(
            (species, coords),
            mm_coords,
            mm_charges,
            charge=net_charge,
            spin_mult=spin_mult,
            mm_cutoff=mm_cutoff,
            pairs_12=pairs_12,
            pairs_13=pairs_13,
            pairs_14=pairs_14,
        )
        return _mlmm_energies_atomic_charges(result, coords)


class _MultipoleMLMM(torch.nn.Module):
    # The model is any module with a 'compute_multipole_mlmm' method
    def __init__(self, model: tp.Any) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        mm_coords: Tensor,
        mm_charges: Tensor,
        mm_dipoles: Tensor,
        mm_quads: Tensor,
        net_charge: Tensor,
        spin_mult: Tensor,
        mm_cutoff: Tensor,
        pairs_12: Tensor,
        pairs_13: Tensor,
        pairs_14: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = self._model.compute_multipole_mlmm(
            (species, coords),
            mm_coords,
            mm_charges,
            mm_dipoles,
            mm_quads,
            charge=net_charge,
            spin_mult=spin_mult,
            mm_cutoff=mm_cutoff,
            pairs_12=pairs_12,
            pairs_13=pairs_13,
            pairs_14=pairs_14,
        )
        return _mlmm_energies_atomic_charges(result, coords)


class _EnergiesForcesAtomicCharges(torch.nn.Module):
    def __init__(self, energies_module: torch.nn.Module) -> None:
        super().__init__()
        self._energies_module = energies_module

    def forward(
        self, species: Tensor, coords: Tensor, *args: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            coords = coords.detach().requires_grad_(True)
            energies, atomic_charges = self._energies_module(species, coords, *args)
            forces = -torch.autograd.grad(energies.sum(), coords)[0]
        return energies.detach(), forces, atomic_charges.detach()


class _EnergiesMLMMForcesAtomicCharges(torch.nn.Module):
    def __init__(self, energies_module: torch.nn.Module) -> None:
        super().__init__()
        self._energies_module = energies_module

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        mm_coords: Tensor,
        mm_charges: Tensor,
        net_charge: Tensor,
        spin_mult: Tensor,
        mm_cutoff: Tensor,
        pairs_12: Tensor,
        pairs_13: Tensor,
        pairs_14: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            coords = coords.detach().requires_grad_(True)
            mm_coords = mm_coords.detach().requires_grad_(True)
            energies, atomic_charges = self._energies_module(
                species,
                coords,
                mm_coords,
                mm_charges,
                net_charge,
                spin_mult,
                mm_cutoff,
                pairs_12,
                pairs_13,
                pairs_14,
            )
            grads = torch.autograd.grad(energies.sum(), (coords, mm_coords))
        forces = -grads[0]
        mm_forces = -grads[1]
        return energies.detach(), forces, mm_forces, atomic_charges.detach()


class _EnergiesMultipoleMLMMForcesAtomicCharges(torch.nn.Module):
    def __init__(self, energies_module: torch.nn.Module) -> None:
        super().__init__()
        self._energies_module = energies_module

    def forward(
        self,
        species: Tensor,
        coords: Tensor,
        mm_coords: Tensor,
        mm_charges: Tensor,
        mm_dipoles: Tensor,
        mm_quads: Tensor,
        net_charge: Tensor,
        spin_mult: Tensor,
        mm_cutoff: Tensor,
        pairs_12: Tensor,
        pairs_13: Tensor,
        pairs_14: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            coords = coords.detach().requires_grad_(True)
            mm_coords = mm_coords.detach().requires_grad_(True)
            energies, atomic_charges = self._energies_module(
                species,
                coords,
                mm_coords,
                mm_charges,
                mm_dipoles,
                mm_quads,
                net_charge,
                spin_mult,
                mm_cutoff,
                pairs_12,
                pairs_13,
                pairs_14,
            )
            grads = torch.autograd.grad(energies.sum(), (coords, mm_coords))
        forces = -grads[0]
        mm_forces = -grads[1]
        return energies.detach(), forces, mm_forces, atomic_charges.detach()


def _example_inputs(
    model: tp.Any,
    atoms: int = 30,
    symbols: tp.Sequence[str] | None = None,
    cutoff: float | None = None,
) -> tuple[Tensor, ...]:
    if symbols is None:
        symbols = model.symbols
    if cutoff is None:
        cutoff = model.cutoff
    param = next(model.parameters())
    gen = torch.Generator().manual_seed(0)
    # Random molecule at roughly liquid density, so it has many pairs and triples
    side = (atoms / 0.1) ** (1 / 3)
    coords = torch.rand(1, atoms, 3, generator=gen, dtype=param.dtype) * side
    cell = torch.eye(3, dtype=param.dtype) * side
    if getattr(model, "periodic_table_index", True):
        elements = torch.tensor([ATOMIC_NUMBER[s] for s in symbols])
    else:
        elements = torch.arange(len(symbols))
    species = elements[torch.randint(0, len(elements), (1, atoms), generator=gen)]
    dist = torch.cdist(coords[0], coords[0])
    is_upper = torch.ones(atoms, atoms, dtype=torch.bool).triu(1)
    neighbor_idxs = ((dist < cutoff) & is_upper).nonzero().t().contiguous()
    shifts = torch.zeros(neighbor_idxs.shape[1], 3, dtype=param.dtype)
    inputs = (species, coords, cell, neighbor_idxs, shifts)
    return tuple(t.to(param.device) for t in inputs)


def _export_energies_forces(
    model: torch.nn.Module,
    entrypoint: str,
    symbols: tp.Sequence[str] | None,
    cutoff: float | None,
) -> ExportedProgram:
    species, coords, cell, neighbor_idxs, shifts = _example_inputs(
        model, symbols=symbols, cutoff=cutoff
    )
    pbc = torch.ones(3, dtype=torch.bool, device=coords.device)
    net_charge = torch.tensor([0], device=coords.device)
    static = Dim.STATIC
    atoms = Dim("atoms", min=2)
    pairs = Dim("pairs", min=2)
    mm_atoms = Dim("mm_atoms", min=2)
    pairs_12_dim = Dim("pairs_12", min=0)
    pairs_13_dim = Dim("pairs_13", min=0)
    pairs_14_dim = Dim("pairs_14", min=0)
    pairs_shapes: tp.List[tp.Tuple[tp.Any, ...]] = [
        (static, pairs_12_dim),
        (static, pairs_13_dim),
        (static, pairs_14_dim),
    ]
    species_coords_shapes: tp.List[tp.Tuple[tp.Any, ...]] = [
        (static, atoms),
        (static, atoms, static),
    ]
    gen = torch.Generator().manual_seed(1)
    # MM charges in the cell next to the ML atoms, so that no pair is too close
    mm_coords = torch.rand(1, 20, 3, generator=gen, dtype=coords.dtype)
    mm_coords[..., 0] += 1.0
    mm_coords = (mm_coords * cell.diagonal().cpu()).to(coords.device)
    mm_charges = torch.rand(1, 20, generator=gen, dtype=coords.dtype) - 0.5
    mm_charges = mm_charges.to(coords.device)
    mm_dipoles = torch.rand(1, 20, 3, generator=gen, dtype=coords.dtype) - 0.5
    mm_dipoles = mm_dipoles.to(coords.device)
    mm_quads = torch.rand(1, 20, 3, 3, generator=gen, dtype=coords.dtype) - 0.5
    mm_quads = mm_quads.to(coords.device)
    spin_mult = torch.tensor([1], device=coords.device)
    mm_cutoff = coords.new_zeros(1)
    pairs_12 = torch.tensor([[0, 1, 2], [1, 2, 3]], device=coords.device)
    pairs_13 = pairs_12.clone()
    pairs_14 = pairs_12.clone()
    module: torch.nn.Module
    forces_wrapper: type[torch.nn.Module] = _EnergiesForcesAtomicCharges
    if entrypoint == "single_point":
        module = _SinglePoint(model)
        args: tuple[Tensor, ...] = (species, coords, net_charge)
        dynamic_shapes = species_coords_shapes + [(static,)]
    elif entrypoint == "single_point_pbc":
        module = _SinglePointPBC(model)
        args = (species, coords, cell, pbc, net_charge)
        dynamic_shapes = species_coords_shapes + [
            (static, static),
            (static,),
            (static,),
        ]
    elif entrypoint == "external_neighbors":
        module = _ExternalNeighbors(model)
        args = (species, coords, neighbor_idxs, shifts, net_charge)
        dynamic_shapes = species_coords_shapes + [
            (static, pairs),
            (pairs, static),
            (static,),
        ]
    elif entrypoint == "custom_mlmm":
        module = _CustomMLMM(model)
        forces_wrapper = _EnergiesMLMMForcesAtomicCharges
        args = (
            species,
            coords,
            mm_coords,
            mm_charges,
            net_charge,
            spin_mult,
            mm_cutoff,
            pairs_12,
            pairs_13,
            pairs_14,
        )
        dynamic_shapes = species_coords_shapes + [
            (static, mm_atoms, static),
            (static, mm_atoms),
            (static,),
            (static,),
            (static,),
        ] + pairs_shapes
    else:
        module = _MultipoleMLMM(model)
        forces_wrapper = _EnergiesMultipoleMLMMForcesAtomicCharges
        args = (
            species,
            coords,
            mm_coords,
            mm_charges,
            mm_dipoles,
            mm_quads,
            net_charge,
            spin_mult,
            mm_cutoff,
            pairs_12,
            pairs_13,
            pairs_14,
        )
        dynamic_shapes = species_coords_shapes + [
            (static, mm_atoms, static),
            (static, mm_atoms),
            (static, mm_atoms, static),
            (static, mm_atoms, static, static),
            (static,),
            (static,),
            (static,),
        ] + pairs_shapes
    # Neighborlist and triple sizes are data-dependent (nonzero, unique, .item())
    with torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True
    ):
        # Strict mode is needed, non-strict can't trace the triples computation
        energies_program = torch.export.export(
            module,
            args,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        # torch.export can't trace autograd.grad (forces). make_fx traces below the
        # autograd layer, so it records the backward ops into the graph. The model
        # buffers are real tensors, so non-fake inputs must be allowed
        energies_forces_graph = make_fx(
            forces_wrapper(energies_program.module()),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(*args)
        return torch.export.export(
            energies_forces_graph, args, dynamic_shapes=dynamic_shapes
        )


def _clear_executable_stack(so_path: str) -> None:
    # Same as 'patchelf --clear-execstack', for 64-bit little-endian ELF files
    _PT_GNU_STACK = 0x6474E551
    _PF_X = 0x1
    with open(so_path, "r+b") as f:
        header = f.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise RuntimeError(f"{so_path} is not a 64-bit little-endian ELF file")
        program_headers_offset = struct.unpack_from("<Q", header, 0x20)[0]
        program_header_size = struct.unpack_from("<H", header, 0x36)[0]
        program_headers_num = struct.unpack_from("<H", header, 0x38)[0]
        for i in range(program_headers_num):
            offset = program_headers_offset + i * program_header_size
            f.seek(offset)
            segment_type, flags = struct.unpack("<II", f.read(8))
            if segment_type == _PT_GNU_STACK:
                f.seek(offset + 4)
                f.write(struct.pack("<I", flags & ~_PF_X))


# Takes around 4 min to compile a *single model*
# For ANI models, a single package with all configurations is around ~170 MB
def compile_model(
    model: torch.nn.Module,
    package_path: str | Path,
    device: Device | tuple[Device, ...] = ("cpu", "cuda"),
    precision: Precision | tuple[Precision, ...] = ("f32", "f64"),
    configs: Config | tuple[Config, ...] | None = None,
    symbols: tp.Sequence[str] | None = None,
    cutoff: float | None = None,
) -> Path:
    r"""Compile a model's energies and forces into an AOTInductor package

    The model is compiled once for each combination of entrypoint, device and
    precision, and each compiled model is stored in the package under the name
    ``"<entrypoint>/<device>/<precision>"``, e.g. ``"single_point_pbc/cuda/f32"``. See
    the module docstring for the signatures of the entrypoints. The input model is not
    modified.

    Args:
        model: The model to compile.
        package_path: Path of the output ``.pt2`` package.
        device: Device, or tuple of devices, to compile for.
        precision: Precision, or tuple of precisions, to compile for.
        configs: Entrypoint, or tuple of entrypoints, to compile. ``"custom_mlmm"``
            requires a model with a ``compute_mlmm`` method, and ``"multipole_mlmm"``
            one with a ``compute_multipole_mlmm`` method. If ``None``,
            ``"single_point"`` and ``"single_point_pbc"`` are always compiled,
            ``"external_neighbors"`` is compiled if the model has a
            ``compute_from_external_neighbors`` method, ``"custom_mlmm"`` if it has
            a ``compute_mlmm`` method, and ``"multipole_mlmm"`` if it has a
            ``compute_multipole_mlmm`` method.
        symbols: Chemical symbols supported by the model, used to build the example
            inputs. If ``None``, ``model.symbols`` is used.
        cutoff: Cutoff of the model, used to build the example inputs. If ``None``,
            ``model.cutoff`` is used.
    Returns:
        The path of the package.
    """
    devices = (device,) if isinstance(device, str) else device
    precisions = (precision,) if isinstance(precision, str) else precision
    has_external_neighbors = hasattr(model, "compute_from_external_neighbors")
    has_custom_mlmm = hasattr(model, "compute_mlmm")
    has_multipole_mlmm = hasattr(model, "compute_multipole_mlmm")
    _configs: tuple[Config, ...]
    if configs is None:
        _configs = ("single_point", "single_point_pbc")
        if has_external_neighbors:
            _configs += ("external_neighbors",)
        if has_custom_mlmm:
            _configs += ("custom_mlmm",)
        if has_multipole_mlmm:
            _configs += ("multipole_mlmm",)
    elif isinstance(configs, str):
        _configs = (configs,)
    else:
        _configs = configs
    if "custom_mlmm" in _configs and not has_custom_mlmm:
        raise ValueError(
            "The 'custom_mlmm' entrypoint requires a model with a 'compute_mlmm' method"
        )
    if "multipole_mlmm" in _configs and not has_multipole_mlmm:
        raise ValueError(
            "The 'multipole_mlmm' entrypoint requires a model with a"
            " 'compute_multipole_mlmm' method"
        )
    if "external_neighbors" in _configs and not has_external_neighbors:
        raise ValueError(
            "The 'external_neighbors' entrypoint requires a model with a"
            " 'compute_from_external_neighbors' method"
        )
    if symbols is None and not hasattr(model, "symbols"):
        raise ValueError("'symbols' must be passed if the model has no 'symbols'")
    if cutoff is None and not hasattr(model, "cutoff"):
        raise ValueError("'cutoff' must be passed if the model has no 'cutoff'")
    # These can't be exported with a dynamic number of atoms (the adaptive lists branch
    # on it, and AllPairs specializes on it). The compiled models only take one
    # structure, which CellList supports
    swappable = (AllPairs, AdaptiveList, FastAdaptiveList)
    owners = [m for m in model.modules() if hasattr(m, "neighborlist")]
    if any(type(owner.neighborlist) in swappable for owner in owners):
        model = copy.deepcopy(model)
        owners = [m for m in model.modules() if hasattr(m, "neighborlist")]
        cell_list = CellList()
        for owner in owners:
            if type(owner.neighborlist) in swappable:
                owner.neighborlist = cell_list
    if any(config != "external_neighbors" for config in _configs):
        for owner in owners:
            if type(owner.neighborlist) not in (CellList, FastCellList):
                raise ValueError(
                    "Only CellList and FastCellList neighborlists can be compiled,"
                    f" got {type(owner.neighborlist).__name__}"
                )
    uses_cell_list = any(type(owner.neighborlist) is CellList for owner in owners)
    aoti_files: dict[str, list[tp.Any]] = {}
    for entrypoint in _configs:
        options = {"aot_inductor.package": True}
        # Autotuning pointwise kernels runs them on made-up data, which triggers
        # out-of-bounds asserts in the indexing kernels of CellList
        uses_neighborlist = entrypoint != "external_neighbors"
        if uses_neighborlist and uses_cell_list:
            options["triton.autotune_pointwise"] = False
        for _device in devices:
            for _precision in precisions:
                dtype = _DTYPES[_precision]
                _model: tp.Any = copy.deepcopy(model).to(device=_device, dtype=dtype)
                # cuAEV is only compiled for CUDA. 'auto' selects it if the model
                # supports it, and otherwise falls back to pyaev
                if hasattr(_model, "set_strategy"):
                    _model.set_strategy("auto" if _device == "cuda" else "pyaev")
                program = _export_energies_forces(_model, entrypoint, symbols, cutoff)
                files = tp.cast(
                    list[str],
                    torch._inductor.aot_compile(
                        program.module(check_guards=False),
                        program.example_inputs[0],
                        options=options,
                    ),
                )
                # The default asm build of the weights produces a shared library that
                # requires an executable stack, which glibc >= 2.41 refuses to load
                for file in files:
                    if file.endswith(".so"):
                        _clear_executable_stack(file)
                name = f"{entrypoint}/{_device}/{_precision}"
                aoti_files[name] = files
    package_aoti(str(package_path), aoti_files)
    return Path(package_path)


__all__ = ["compile_model"]
