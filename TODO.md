
What is needed to support `torch.compile` with fullgraph=True

Issues with cuAEV:
- The ATen impl (`aev_cpu.cpp`, registered to CPU and AutogradCPU) is not opaque under tracing
  and has data-dependent ops (`nonzero`, `.item()` for `max_count`, `tril_indices`), with no
  compile-specific paths (unlike `neighbors_to_triples` in python). Compiling cuAEV on CPU
  currently raises an error from `_prepare_cuaev_execution`

cuAEV op registration, compared with current PyTorch recommendations:
- Library loading (`torch.ops.load_library` vs import). Not a bug, only a packaging choice.
  - `torch.ops.load_library` is not deprecated. It is the documented way to load libraries that
    register ops but are not Python extension modules (no `PyInit_*`), e.g. libraries built with
    CMake for libtorch, or with `cpp_extension.load(is_python_module=False)`. All 3 builds of
    `cuaev` are of this kind (`setup.py` with `no_python_abi_suffix=True`, `ani build-extensions`,
    and `CMakeLists.txt`), so `import torchani.cuaev` doesn't work and `load_library` is required.
  - The current custom ops tutorial instead adds a dummy `PyInit__C` to the C++ code, builds with
    setuptools (optionally `py_limited_api=True`) and imports the module to run the registrations.
  - Advantages of importing: Python finds the file (no hand written internal/external paths and
    `.so`/`.pyd`/`.dll` suffixes in `torchani/csrc/__init__.py`), one `abi3` wheel works for
    several Python versions with `py_limited_api`, and failures are standard `ImportError`s.
  - Disadvantages: `PyInit_*` references the Python C API, so loading the library from pure C++
    (LAMMPS) needs care. On Linux lazy binding usually tolerates it, on Windows the DLL would
    depend on `python3X.dll`. The CMake and `ani build-extensions` builds would probably still
    be loaded by path.
  - Registration behaves the same either way.
- Host syncs and the CPU `sizes` output. The CUDA ops can't be captured in CUDA graphs (e.g.
  `torch.compile(mode="reduce-overhead")`, not tested).
  - The forward needs some data dependent values on the host: `nI` (number of atoms with
    neighbors) is the grid size of the radial and angular kernels, `maxNumJPerI` sets their
    dynamic shared memory size, and in the fused op the radial `nJ` sets the size of the nbrlist
    buffers. These are copied to the host with `cudaMemcpyAsync` and `cudaStreamSynchronize`
    (`cubDeviceSelectIf`, `cubEncode`, `cubMax`) or `.item()`. In release builds this gives
    1 sync per forward for the full nbrlist, 2 for the half nbrlist and for a single molecule
    with the fused op, and 3 for a batch with the fused op.
  - The backward kernels need `nI` and `maxNumJPerI` for the same reasons, so the forward returns
    them in a CPU `sizes` tensor and the backward doesn't need to sync again. In release builds
    the `nJ` entries of `sizes` are only computed by the fused op (the half and full nbrlists
    leave them as 0), the backward doesn't use them.
  - Removing the syncs is feasible but is a substantial kernel change:
    - `nI`: launch `total_atoms` blocks (an upper bound) and exit early in blocks with an index
      `>= nI`, read from device memory. Relatively simple, idle blocks only cost much with lots of
      padding.
    - `maxNumJPerI`: the angular shared memory is `(angular_len + 6 * maxNumJPerI)` values per
      block. Using an upper bound (`MAX_NUMJ_PER_I_IN_RCR = 1000`) gives ~28 KB per block in
      float32 and ~56 KB in float64 (above the default 48 KB limit), versus a few KB with the
      usual ~60 neighbors, which lowers occupancy. Avoiding that requires rewriting the kernels to
      loop over neighbors in fixed size tiles.
    - Fused `nJ`: allocate the buffers at the upper bound `n_atoms * min(n_atoms, 1000)` per
      molecule (~40 B per pair kept for the backward, ~400 MB at 10k atoms), or build the nbrlist
      with the half nbrlist path.
    - Once no values are needed on the host, `sizes` can stay on the device.
  - A cheaper partial change is to merge syncs (e.g. in the batched fused op), which reduces
    latency for small systems but still doesn't allow CUDA graph capture.

Issues with `cell_list` and `mnp`
These can be registered the same way as with the `cuAEV`

Build:
- `cuaev` is only built as a CUDA extension. `aev_cpu.cpp` has no CUDA dependency, so a CPU-only
  build only needs the sources split by backend in `setup.py`, `cli.py` and `CMakeLists.txt`.

Cleanup:
- Unused args: `coords` in `AEVComputer.compute_from_neighbors` (unused by both pyaev and
  cuAEV), `distances` in `run_with_half_nbrlist_aten`.
- Generated docs in `docs/build/` still show the old `CuaevComputer` code.
