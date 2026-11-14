r"""Fake kernels for the cuAEV ops, needed to trace them (e.g. with torch.compile)"""

import torch
from torch import Tensor


def _aev_len(
    eta_r: Tensor,
    shf_r: Tensor,
    eta_a: Tensor,
    zeta: Tensor,
    shf_a: Tensor,
    shf_z: Tensor,
    num_species: int,
) -> int:
    radial_sublen = eta_r.shape[0] * shf_r.shape[0]
    angular_sublen = eta_a.shape[0] * zeta.shape[0] * shf_a.shape[0] * shf_z.shape[0]
    num_species_pairs = num_species * (num_species + 1) // 2
    return radial_sublen * num_species + angular_sublen * num_species_pairs


# Must match the layout of the outputs of the forward ops in cuaev.cpp. The CPU impl
# only returns the aev
def _forward_outputs(
    species: Tensor,
    like: Tensor,
    aev_len: int,
    num_pairs: int | torch.SymInt,
    num_pair_idxs: int,
) -> list[Tensor]:
    num_molecs, num_atoms = species.shape
    total_atoms = num_molecs * num_atoms
    aev = like.new_empty((num_molecs, num_atoms, aev_len))
    if like.device.type == "cpu":
        return [aev]
    outputs = [
        aev,
        torch.empty(5, dtype=torch.int64, device="cpu"),
        like.new_empty(total_atoms * 2, dtype=torch.int32),
        like.new_empty(total_atoms, dtype=torch.int32),
    ]
    for _ in range(2):
        outputs.append(like.new_empty(num_pairs, dtype=torch.int32))
        outputs.append(like.new_empty(total_atoms, dtype=torch.int32))
        outputs.append(like.new_empty(num_pairs))
        outputs.append(like.new_empty(num_pairs * 3))
        outputs.append(like.new_empty(num_pair_idxs, dtype=torch.int32))
    return outputs


@torch.library.register_fake("cuaev::run")
def _run(
    coords,
    species,
    rcr,
    rca,
    eta_r,
    shf_r,
    eta_a,
    zeta,
    shf_a,
    shf_z,
    num_species,
    use_cos,
):
    aev_len = _aev_len(eta_r, shf_r, eta_a, zeta, shf_a, shf_z, num_species)
    # The number of pairs within the cutoff is only known after running the kernels
    num_pairs = torch.library.get_ctx().new_dynamic_size()
    return _forward_outputs(species, coords, aev_len, num_pairs, 0)


@torch.library.register_fake("cuaev::run_with_half_nbrlist")
def _run_with_half_nbrlist(
    species,
    atom_ij,
    delta_j,
    dist_j,
    rcr,
    rca,
    eta_r,
    shf_r,
    eta_a,
    zeta,
    shf_a,
    shf_z,
    num_species,
    use_cos,
):
    aev_len = _aev_len(eta_r, shf_r, eta_a, zeta, shf_a, shf_z, num_species)
    num_pairs = atom_ij.shape[1] * 2
    return _forward_outputs(species, delta_j, aev_len, num_pairs, num_pairs)


@torch.library.register_fake("cuaev::run_with_full_nbrlist")
def _run_with_full_nbrlist(
    coords,
    species,
    atom_i,
    atom_j,
    num_j_per_i,
    rcr,
    rca,
    eta_r,
    shf_r,
    eta_a,
    zeta,
    shf_a,
    shf_z,
    num_species,
    use_cos,
):
    aev_len = _aev_len(eta_r, shf_r, eta_a, zeta, shf_a, shf_z, num_species)
    return _forward_outputs(species, coords, aev_len, atom_j.shape[0], 0)


@torch.library.register_fake("cuaev::_backward")
def _backward(
    grad_aev,
    grad_target,
    species,
    saved,
    rcr,
    rca,
    eta_r,
    shf_r,
    eta_a,
    zeta,
    shf_a,
    shf_z,
    num_species,
    use_cos,
):
    return grad_target.new_empty(grad_target.shape)


@torch.library.register_fake("cuaev::_double_backward")
def _double_backward(
    grad_force,
    grad_target,
    species,
    saved,
    rcr,
    rca,
    eta_r,
    shf_r,
    eta_a,
    zeta,
    shf_a,
    shf_z,
    num_species,
    use_cos,
):
    aev_len = _aev_len(eta_r, shf_r, eta_a, zeta, shf_a, shf_z, num_species)
    return grad_target.new_empty((species.shape[0], species.shape[1], aev_len))
