import typing as tp

import torch
from torch import Tensor

from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE


# Module that supports a sequence of chemical symbols
class _ChemModule(torch.nn.Module):

    atomic_numbers: Tensor
    _conv_tensor: Tensor

    def __init__(self, symbols: tp.Sequence[str] = ()) -> None:
        super().__init__()
        atomic_numbers = torch.tensor(
            [ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long
        )
        # First element is extra-pair and last element will always be -1
        conv_tensor = -torch.ones(118 + 2, dtype=torch.long)
        for i, znum in enumerate(atomic_numbers):
            conv_tensor[znum] = i

        self.register_buffer("atomic_numbers", atomic_numbers, persistent=False)
        self.register_buffer("_conv_tensor", conv_tensor, persistent=False)

    # Validate a seqof floats that must have the same len as the provided symbols
    # default must be a mapping or sequence that when indexed with
    # atomic numbers returns floats
    @torch.jit.unused
    def _validate_elem_seq(
        self,
        name: str,
        seq: tp.Sequence[float],
        default: tp.Sequence[float] = (),
        pair: bool = False,
    ) -> tp.Sequence[float]:
        if not pair:
            if not seq and default:
                seq = [default[j] for j in self.atomic_numbers]

        if not all(isinstance(v, float) for v in seq):
            raise ValueError(f"Some values in {name} are not floats")
        num_elem = len(self.symbols)
        num_expect = num_elem if not pair else num_elem * (num_elem + 1) // 2
        if not len(seq) == num_expect:
            if not pair:
                raise ValueError(f"{name} and symbols should have the same len")
            else:
                raise ValueError(
                    f"{name} should have len (num-symbols * (num-symbols + 1) / 2)"
                )
        return seq

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)
