#include <torch/all.h>

// Explicit schemas also keep the argument names in python
TORCH_LIBRARY(cuaev, m) {
  m.set_python_module("torchani.csrc._cuaev_fake");
  // The kernels read the inputs through raw pointers, so they must be contiguous
  const std::vector<at::Tag> tags = {at::Tag::needs_contiguous_strides, at::Tag::pt2_compliant_tag};
  m.def(
      "run(Tensor coordinates, Tensor species, float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, "
      "Tensor Zeta, Tensor ShfA, Tensor ShfZ, int num_species, bool use_cos_cutoff) -> Tensor[]",
      tags);
  m.def(
      "run_with_half_nbrlist(Tensor species, Tensor atomIJ, Tensor deltaJ, Tensor distJ, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor[]",
      tags);
  m.def(
      "run_with_full_nbrlist(Tensor coordinates, Tensor species, Tensor atomI, Tensor atomJ, Tensor numJPerI, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor[]",
      tags);
  // Internal ops called by the autograd functions
  m.def(
      "_backward(Tensor grad_aev, Tensor grad_target, Tensor species, Tensor[] saved, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor",
      tags);
  m.def(
      "_double_backward(Tensor grad_force, Tensor grad_target, Tensor species, Tensor[] saved, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor",
      tags);
}
