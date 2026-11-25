r"""Fake kernels for the cell list ops, needed for torch.compile and export"""

import torch
from torch import Tensor


@torch.library.register_fake("cell_list::cell_list")
def _cell_list(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    # The number of pairs within the cutoff is only known after running the op
    num_pairs = torch.library.get_ctx().new_dynamic_size()
    neighbor_idxs = coords.new_empty((2, num_pairs), dtype=torch.long)
    distances = coords.new_empty(num_pairs)
    diff_vectors = coords.new_empty((num_pairs, 3))
    return neighbor_idxs, distances, diff_vectors
