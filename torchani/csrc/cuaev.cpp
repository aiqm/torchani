#include <aev.h>
#include <torch/extension.h>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

AEVScalarParams::AEVScalarParams(
    float Rcr,
    float Rca,
    Tensor EtaR_t,
    Tensor ShfR_t,
    Tensor EtaA_t,
    Tensor Zeta_t,
    Tensor ShfA_t,
    Tensor ShfZ_t,
    int num_species,
    bool use_cos_cutoff)
    : Rcr(Rcr),
      Rca(Rca),
      radial_sublength(EtaR_t.size(0) * ShfR_t.size(0)),
      angular_sublength(EtaA_t.size(0) * Zeta_t.size(0) * ShfA_t.size(0) * ShfZ_t.size(0)),
      num_species(num_species),
      EtaR_t(EtaR_t),
      ShfR_t(ShfR_t),
      EtaA_t(EtaA_t),
      Zeta_t(Zeta_t),
      ShfA_t(ShfA_t),
      ShfZ_t(ShfZ_t),
      use_cos_cutoff(use_cos_cutoff) {
  radial_length = radial_sublength * num_species;
  angular_length = angular_sublength * (num_species * (num_species + 1) / 2);
}

Result::Result(
    Tensor aev_t,
    Tensor atomI_t,
    Tensor startIdxJ_t,
    int64_t nI,
    Tensor coordinates_t,
    Tensor species_t,
    NeighborList radialNbr,
    NeighborList angularNbr)
    : aev_t(aev_t),
      atomI_t(atomI_t),
      startIdxJ_t(startIdxJ_t),
      nI(nI),
      coordinates_t(coordinates_t),
      species_t(species_t),
      radialNbr(radialNbr),
      angularNbr(angularNbr) {}

Result::Result(tensor_list tensors)
    : aev_t(tensors[0]), // aev_t will be a undefined tensor
      atomI_t(tensors[1]),
      startIdxJ_t(tensors[2]),
      nI(tensors[3].item<int>()),
      coordinates_t(tensors[4]),
      species_t(tensors[5]),
      radialNbr(tensors[6].item<int>(), tensors[7].item<int>(), tensors[8], tensors[9], tensors[10]),
      angularNbr(tensors[11].item<int>(), tensors[12].item<int>(), tensors[13], tensors[14], tensors[15]) {}

Result::Result()
    : aev_t(Tensor()),
      atomI_t(Tensor()),
      startIdxJ_t(Tensor()),
      nI(0),
      coordinates_t(Tensor()),
      species_t(Tensor()),
      radialNbr(NeighborList()),
      angularNbr(NeighborList()) {}

Result::Result(Tensor coordinates_t, Tensor species_t)
    : aev_t(Tensor()),
      atomI_t(Tensor()),
      startIdxJ_t(Tensor()),
      nI(0),
      coordinates_t(coordinates_t),
      species_t(species_t),
      radialNbr(NeighborList()),
      angularNbr(NeighborList()) {}

CuaevComputer::CuaevComputer(
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff)
    : aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff) {}

Tensor CuaevDoubleAutograd::forward(
    AutogradContext* ctx,
    Tensor grad_e_aev,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer,
    tensor_list result_tensors) {
  Tensor grad_coord = cuaev_computer->backward(grad_e_aev, result_tensors);

  if (grad_e_aev.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->save_for_backward(result_tensors);
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0];
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force, ctx->get_saved_variables());
  return {grad_grad_aev, torch::Tensor(), torch::Tensor()};
}

Tensor CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  at::AutoDispatchBelowADInplaceOrView guard;
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  if (coordinates_t.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->save_for_backward(result);
  }
  return result.aev_t;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], cuaev_computer, result_tensors);
  return {grad_coord, Tensor(), Tensor()};
}

Tensor run_only_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  return result.aev_t;
}

Tensor run_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  return CuaevAutograd::apply(coordinates_t, species_t, cuaev_computer);
}

TORCH_LIBRARY(cuaev, m) {
  m.class_<CuaevComputer>("CuaevComputer")
      .def(torch::init<double, double, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, bool>())
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CuaevComputer>& self) -> std::vector<Tensor> {
            std::vector<Tensor> state;
            state.push_back(torch::tensor(self->aev_params.Rcr));
            state.push_back(torch::tensor(self->aev_params.Rca));
            state.push_back(self->aev_params.EtaR_t);
            state.push_back(self->aev_params.ShfR_t);
            state.push_back(self->aev_params.EtaA_t);
            state.push_back(self->aev_params.Zeta_t);
            state.push_back(self->aev_params.ShfA_t);
            state.push_back(self->aev_params.ShfZ_t);
            state.push_back(torch::tensor(self->aev_params.num_species));
            state.push_back(torch::tensor(self->aev_params.use_cos_cutoff));
            return state;
          },
          // __setstate__
          [](std::vector<Tensor> state) -> c10::intrusive_ptr<CuaevComputer> {
            return c10::make_intrusive<CuaevComputer>(
                state[0].item<double>(),
                state[1].item<double>(),
                state[2],
                state[3],
                state[4],
                state[5],
                state[6],
                state[7],
                state[8].item<int64_t>(),
                state[9].item<bool>());
          });
  m.def("run", run_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("run", run_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", run_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
