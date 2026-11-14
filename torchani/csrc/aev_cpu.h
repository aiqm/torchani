#pragma once

#include <torch/extension.h>

// AEV implementation written with differentiable ATen ops. It works for tensors in any device.
// It is registered as the CPU impl of the cuaev ops, and it is also used to compute higher order
// derivatives that the CUDA kernels don't support.

torch::Tensor run_aten(
    const torch::Tensor& coordinates,
    const torch::Tensor& species,
    double Rcr,
    double Rca,
    const torch::Tensor& EtaR,
    const torch::Tensor& ShfR,
    const torch::Tensor& EtaA,
    const torch::Tensor& Zeta,
    const torch::Tensor& ShfA,
    const torch::Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff);

torch::Tensor run_with_half_nbrlist_aten(
    const torch::Tensor& species,
    const torch::Tensor& atomIJ,
    const torch::Tensor& diff_vectors,
    const torch::Tensor& distances,
    double Rcr,
    double Rca,
    const torch::Tensor& EtaR,
    const torch::Tensor& ShfR,
    const torch::Tensor& EtaA,
    const torch::Tensor& Zeta,
    const torch::Tensor& ShfA,
    const torch::Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff);

torch::Tensor run_with_full_nbrlist_aten(
    const torch::Tensor& coordinates,
    const torch::Tensor& species,
    const torch::Tensor& atomI,
    const torch::Tensor& atomJ,
    const torch::Tensor& numJPerI,
    double Rcr,
    double Rca,
    const torch::Tensor& EtaR,
    const torch::Tensor& ShfR,
    const torch::Tensor& EtaA,
    const torch::Tensor& Zeta,
    const torch::Tensor& ShfA,
    const torch::Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff);
