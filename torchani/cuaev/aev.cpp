#include <torch/extension.h>

template <typename ScalarRealT = float>
torch::Tensor cuComputeAEV(torch::Tensor coordinates_t, torch::Tensor species_t,
                           double Rcr_, double Rca_, torch::Tensor EtaR_t,
                           torch::Tensor ShfR_t, torch::Tensor EtaA_t,
                           torch::Tensor Zeta_t, torch::Tensor ShfA_t,
                           torch::Tensor ShfZ_t, int64_t num_species_) {
  ScalarRealT Rcr = Rcr_;
  ScalarRealT Rca = Rca_;
  int num_species = num_species_;
}

TORCH_LIBRARY(cuaev, m) { m.def("cuComputeAEV", &cuComputeAEV<float>); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
