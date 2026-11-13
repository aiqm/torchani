
What is needed to support `torch.compile` with fullgraph=True


- `neighbors_to_triples` currently fails. The offending op is [DONE]
  `torch.tril_indices(counts_max, counts_max, -1)` it needs to be replaced with
    `ar = torch.arange(counts_max); (ar.view(-1, 1) > ar.view(1, -1).nonzero().t())`

- `if selected_idxs.size(0) > 0` branches on data dependent size, but since the empty [DONE]
    branch works fine the guard can be skipped when compiling

- There is an existing bug in `narrow_down` [DONE]
    Skipping when non compiling is a bad idea, better to always remove padding atoms
    when compiling, and don't perform the check.

- `PBC with allpairs on CUDA` the shift counts come from .item() calls
    and products make inductor fail. There is a workaround but it is bad.
    it should be possible to fix this by removing the data-dependent sizes
    This needs specific hints for dynamo

- Inductor has a bug for CPU code, it requires `scatter_add` instead of `index_add`


CellList required a bunch of patches to pass (very minor, things that branch on tensor
data and he `tril_indices` patch)

Issues with cuAEV:
- Fails to trace because it needs torch-compile specific things
1. Removing `torch.classes.cuaev.CuaevComputer` altogheter
2. `fake kernels` for `cuaev::run` `run_with_half_nbrlist` and `run_with_full_nbrlist`
3. register the bwd, double bwd as ops and connect them with
   `torch.library.register_autograd`  instead of C++ directly.
  
Issues with `cell_list` and `mnp`
These can be regisered the same way as with the `cuAEV`

Users must set:
- `torch._dyano.config.capture_dynamic_output_shape_ops = True`
- `torch._dyano.config.capture_scalar_outputs = True`
This is because of `nonzero`, `unique`, and `item()`

