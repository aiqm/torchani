r"""Ahead-of-time compilation of models into AOTInductor packages

.. warning::

    This module is experimental. Its API and behavior may change without warning.

The compiled package computes energies and forces from an external neighborlist, and
can be loaded without Python, from C++, using
``torch::inductor::AOTIModelPackageLoader``, or from Python, using
``torch._inductor.aoti_load_package``. Its signature is:

``(species, coords, neighbor_idxs, shifts) -> (energies, forces)``

with shapes ``(1, A)``, ``(1, A, 3)``, ``(2, P)`` and ``(P, 3)``. The number of atoms
``A`` and of pairs ``P`` are dynamic. All inputs must be contiguous, and ``coords``
must be mapped to the central cell. If the model uses cuAEV, the cuAEV extension must
be loaded by the process before running the package.
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.export import Dim
from torch.fx.experimental.proxy_tensor import make_fx

from torchani.arch import _ANI
from torchani.constants import ATOMIC_NUMBER


class _Energies(torch.nn.Module):
    def __init__(self, model: _ANI) -> None:
        super().__init__()
        self._model = model

    def forward(
        self, species: Tensor, coords: Tensor, neighbor_idxs: Tensor, shifts: Tensor
    ) -> Tensor:
        return self._model.compute_from_external_neighbors(
            species, coords, neighbor_idxs, shifts
        ).energies


class _EnergiesForces(torch.nn.Module):
    def __init__(self, energies_module: torch.nn.Module) -> None:
        super().__init__()
        self._energies_module = energies_module

    def forward(
        self, species: Tensor, coords: Tensor, neighbor_idxs: Tensor, shifts: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            coords = coords.detach().requires_grad_(True)
            energies = self._energies_module(species, coords, neighbor_idxs, shifts)
            forces = -torch.autograd.grad(energies.sum(), coords)[0]
        return energies.detach(), forces


def _example_inputs(model: _ANI, atoms: int = 30) -> tuple[Tensor, ...]:
    param = next(model.parameters())
    gen = torch.Generator().manual_seed(0)
    # Random molecule at roughly liquid density, so it has many pairs and triples
    side = (atoms / 0.1) ** (1 / 3)
    coords = torch.rand(1, atoms, 3, generator=gen, dtype=param.dtype) * side
    if model.periodic_table_index:
        elements = torch.tensor([ATOMIC_NUMBER[s] for s in model.symbols])
    else:
        elements = torch.arange(len(model.symbols))
    species = elements[torch.randint(0, len(elements), (1, atoms), generator=gen)]
    dist = torch.cdist(coords[0], coords[0])
    is_upper = torch.ones(atoms, atoms, dtype=torch.bool).triu(1)
    neighbor_idxs = ((dist < model.cutoff) & is_upper).nonzero().t().contiguous()
    shifts = torch.zeros(neighbor_idxs.shape[1], 3, dtype=param.dtype)
    return tuple(t.to(param.device) for t in (species, coords, neighbor_idxs, shifts))


def compile_model(model: _ANI, package_path: str | Path) -> Path:
    r"""Compile a model's energies and forces into an AOTInductor package

    The model is compiled for the device and dtype of its parameters. See the
    module docstring for the signature of the compiled package.

    Args:
        model: The model to compile.
        package_path: Path of the output ``.pt2`` package.
    Returns:
        The path of the package.
    """
    args = _example_inputs(model)
    static = Dim.STATIC
    atoms = Dim("atoms", min=2)
    pairs = Dim("pairs", min=2)
    dynamic_shapes = [
        (static, atoms),
        (static, atoms, static),
        (static, pairs),
        (pairs, static),
    ]
    # Neighborlist and triple sizes are data-dependent (nonzero, unique, .item())
    with torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True
    ):
        # Strict mode is needed, non-strict can't trace the triples computation
        energies_program = torch.export.export(
            _Energies(model), args, dynamic_shapes=dynamic_shapes, strict=True
        )
        # torch.export can't trace autograd.grad (forces). make_fx traces below the
        # autograd layer, so it records the backward ops into the graph. The model
        # buffers are real tensors, so non-fake inputs must be allowed
        energies_forces_graph = make_fx(
            _EnergiesForces(energies_program.module()),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(*args)
        program = torch.export.export(
            energies_forces_graph, args, dynamic_shapes=dynamic_shapes
        )
    # The default asm build of the weights produces a shared library that requires
    # an executable stack, which glibc >= 2.41 refuses to load
    with torch._inductor.config.patch({"aot_inductor.use_consts_asm_build": False}):
        path = torch._inductor.aoti_compile_and_package(
            program, package_path=str(package_path)
        )
    return Path(path)


__all__ = ["compile_model"]
