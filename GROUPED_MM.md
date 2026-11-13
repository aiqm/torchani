# Grouped matmul evaluation of atomic networks (investigation notes)

## Motivation

`ANINetworks.forward` loops over elements, selecting atoms with `nonzero`. For an
ANI-2x ensemble this means 56 small MLPs per call (7 elements x 8 members), 56
host-device syncs, and ~414 CUDA kernel launches. The networks are CPU / launch
bound (at 8 atoms the GPU is busy ~0.56 ms per call while a call takes > 4 ms), and
`torch.compile` does not fix this: the `nonzero` syncs remain, each unbacked size
splits the graph, and Inductor emits every matmul as a separate extern call.
CUDA graphs (`mode="reduce-overhead"`) cannot be used while there are
data-dependent shapes.

The idea: evaluate all element networks of all members with a few grouped
matmuls, with tensor shapes that depend only on the total number of atoms `N`.

## Schematic of the implementation

Notation: `S` elements, `E` ensemble members, `N` atoms (flattened batch).
Dummy atoms (`elem_idx == -1`) are mapped to an extra group `S` with all-zero
weights.

Weights (built once from an `Ensemble` of `ANINetworks`, frozen):

- Layer widths are zero-padded to the widest element network, and to multiples
  of 4 (16-byte stride requirement of `_grouped_mm`). This is exact because
  padded neurons have zero weights and biases, and `act(0) == 0` for CELU/GELU.
  The final layer (width 1) is padded to 4 and column 0 is used.
- First layer: groups are elements, members concatenated along the output dim.
  `W0: (S + 1, in, E * out)`, `b0: (S + 1, E * out)`.
- Other layers: one group per (element, member) pair, group id `s * E + e`.
  `Wl: ((S + 1) * E, in, out)`, `bl: ((S + 1) * E, out)`.

Forward (no syncs, all shapes depend only on `N`):

```python
groups = torch.where(elem_idxs == -1, S, elem_idxs).flatten()
order = torch.argsort(groups)
sorted_groups = groups[order]
counts = zeros(S + 1).scatter_add(0, groups, ones)  # not bincount, it syncs
ends = cumsum(counts)
starts = ends - counts

# Layer 0: (N, 1008) -> (N, E * H0)
x = aevs.flatten(0, 1)[order]
x = act(grouped_mm(x, W0, offs=ends) + b0[sorted_groups])

# Reorder rows to (element, member, atom) so every (s, e) group is contiguous
local = arange(N) - starts[sorted_groups]
rows = starts[sorted_groups, None] * E + arange(E)[None] * counts[sorted_groups, None] + local[:, None]
x = zeros_like(...).index_copy(0, rows.flatten(), x.view(N * E, H0))

group_counts = counts.repeat_interleave(E)
group_ends = cumsum(group_counts)
row_groups = repeat_interleave(arange((S + 1) * E), group_counts, output_size=N * E)
for l in 1..L-1:
    x = grouped_mm(x, Wl, offs=group_ends) + bl[row_groups]
    x = act(x) if l is not last

# Back to atom order, mean over members, sum over atoms
per_atom = x[:, 0][rows.flatten()].view(N, E).mean(1)
energies = zeros(N).index_copy(0, order, per_atom).view(molecs, atoms).sum(1)
```

`grouped_mm` must be wrapped in a `torch.library.custom_op` (see findings), with
a fake impl returning `(mat_a.shape[0], mat_b.shape[2])` and an autograd formula
`grad_a = grouped_mm(grad.contiguous(), mat_b.transpose(1, 2).contiguous(), offs)`
(frozen weights, so no `grad_b`).

## Findings (torch 2.13.0+cu130, RTX 4090 Laptop, compute capability 8.9)

What works:

- Results match `Ensemble` (ANI-2x, padded batch with all 7 elements) on CUDA
  and CPU: relative error ~1e-7 energies, ~3.5e-6 AEV gradients, ~5e-5 double
  gradients (eager), and the same for energies and gradients when compiled with
  `fullgraph=True`.
- `torch._grouped_mm` runs in float32 in eager mode on CUDA and CPU, including
  backward and double backward, and handles groups with zero rows.

Problems:

- On compute capability < 9.0 `torch._grouped_mm` is **not** a fused kernel, in
  float32 or bf16: it copies `offs` to the host (one sync) and runs one `aten::mm`
  per group. Layers 2+ have `(S + 1) * E = 64` groups, i.e. *more* matmul launches
  than the current per-element loop. Inductor only uses its Triton grouped-mm
  kernel for capability >= 9.0 (`torch/_inductor/kernel/mm_grouped.py`). Whether
  eager `_grouped_mm` is fused on 9.0+ GPUs or on sm_120 was not verified.
- Because of that host copy, `_grouped_mm` cannot be captured in a CUDA graph
  ("Cannot copy between CPU and CUDA tensors during CUDA graph capture").
- The meta (fake) kernel of `_grouped_mm` only accepts bf16, so float32 cannot be
  traced by Dynamo directly; the `custom_op` wrapper is required.
- Backward fails if the incoming gradient is not contiguous (e.g. the expanded
  gradient produced by `.sum()`: "Invalid strides/sizes, got [0, 0]").
- Output widths must be multiples of 4 floats ("strides should be multiple of 16
  bytes"), which is why the width-1 final layer needs padding.
- Weights: 82.8 MiB vs 52.3 MiB for the ANI-2x `Ensemble`, due to padding to the
  widest element network and the extra dummy group.
- `torch.bincount` must not be used for the counts: on CUDA it syncs to find its
  output size.
- Double backward through a compiled module fails, for the existing `Ensemble`
  too ("non-empty donated buffers which requires create_graph=False"), so this is
  a `torch.compile` limitation and not specific to this approach.

Not measured: speed and peak activation memory vs the current eager and compiled
code (the benchmark was stopped before producing results).

## Alternatives considered

- Dense masked evaluation: run every element network on every atom and mask.
  No syncs, so CUDA graphs become possible, but the work is `S` times larger
  (4x for ANI-1x, 7x for ANI-2x), independently of which elements are present.
- Padded `bmm` with a fixed per-element capacity `C`: static shapes require
  `C = N` (same waste as dense), and smaller capacities need user-provided bounds
  or recompilation buckets, plus overflow handling.
- `BmmEnsemble` (`torchani/nn/_infer.py`) batches across ensemble members (7
  batched MLPs instead of 56), but caches per-element `nonzero` indices in module
  state and checks the cache with `.item()`, which is not compile friendly.
