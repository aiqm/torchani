
What is needed to support `torch.compile` with fullgraph=True

Issues with cuAEV:
- Fails to trace because it needs torch-compile specific things
1. Removing `torch.classes.cuaev.CuaevComputer` altogether [done]
2. `fake kernels` for `cuaev::run` `run_with_half_nbrlist` and `run_with_full_nbrlist`
   (currently Meta has no kernel, only the backend fallback)
3. register the bwd, double bwd as ops and connect them with
   `torch.library.register_autograd`  instead of C++ directly.
   - The CUDA autograd kernels call `aev_forward*` / `aev_backward*` directly instead of
     redispatching, so keys below Autograd (Functionalize, Python, FakeTensor) never see the op
   - `Result` is saved as a `tensor_list` with `nI`, `nJ`, `maxNumJPerI` packed as scalar
     tensors and read back with `.item()` (host sync, data-dependent sizes)
4. The ATen impl (`aev_cpu.cpp`, registered to CPU and AutogradCPU) is not opaque under tracing
   and has data-dependent ops (`nonzero`, `.item()` for `max_count`, `tril_indices`), with no
   compile-specific paths (unlike `neighbors_to_triples` in python)
  
Issues with `cell_list` and `mnp`
These can be registered the same way as with the `cuAEV`

Users must set:
- `torch._dynamo.config.capture_dynamic_output_shape_ops = True`
- `torch._dynamo.config.capture_scalar_outputs = True`
This is because of `nonzero`, `unique`, and `item()`

Other cuAEV issues (after removing `CuaevComputer`)

Breaking changes:
- TorchScript models saved with the old extension can't be loaded (they reference
  `torch.classes.cuaev.CuaevComputer` and the old `cuaev::run*` schemas). Pickled
  `AEVComputer` instances from before the change can't be unpickled either. State dicts load fine.
- `cuaev::run_with_half_nbrlist` no longer takes `coordinates` (gradients flow through
  `deltaJ`), external callers with the old signature break.

Build:
- `cuaev` is only built as a CUDA extension. `aev_cpu.cpp` has no CUDA dependency, so a CPU-only
  build only needs the sources split by backend in `setup.py`, `cli.py` and `CMakeLists.txt`.

Performance:
- The CPU (ATen) impl and the new half nbrlist backward (per-pair scatter) are not benchmarked.
- The coordinate Hessian term on CUDA is recomputed with the ATen impl (roughly pyaev speed).
  Force training that uses plain `loss.backward()` with leaf coordinates pays this every step,
  `loss.backward(inputs=params)` avoids it (done in `examples/training.py` and
  `tools/training-aev-benchmark.py` only).

Limitations:
- Third and higher order derivatives wrt coordinates are not supported (the Hessian term is
  computed from detached tensors, so higher order contributions are missing).
- The half nbrlist op requires contiguous `deltaJ` and `distJ` (raises otherwise), could call
  `.contiguous()` in `_cuaev_compute_from_neighbors`.

Cleanup:
- Unused args: `species` in `_prepare_cuaev_execution`, `coords` in
  `AEVComputer.compute_from_neighbors` (unused by both pyaev and cuAEV), `distances` in
  `run_with_half_nbrlist_aten`.
- Generated docs in `docs/build/` still show the old `CuaevComputer` code.
