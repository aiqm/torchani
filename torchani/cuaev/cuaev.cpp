#include <aev.h>
#include <torch/extension.h>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

Result::Result() {
  aev_t = Tensor();
  tensor_Rij = Tensor();
  tensor_radialRij = Tensor();
  tensor_angularRij = Tensor();
  total_natom_pairs = 0;
  nRadialRij = 0;
  nAngularRij = 0;
  tensor_centralAtom = Tensor();
  tensor_numPairsPerCenterAtom = Tensor();
  tensor_centerAtomStartIdx = Tensor();
  maxnbrs_per_atom_aligned = 0;
  angular_length_aligned = 0;
  ncenter_atoms = 0;
  coordinates_t = Tensor();
  species_t = Tensor();
}

Result::Result(
    Tensor aev_t_,
    Tensor tensor_Rij_,
    Tensor tensor_radialRij_,
    Tensor tensor_angularRij_,
    int64_t total_natom_pairs_,
    int64_t nRadialRij_,
    int64_t nAngularRij_,
    Tensor tensor_centralAtom_,
    Tensor tensor_numPairsPerCenterAtom_,
    Tensor tensor_centerAtomStartIdx_,
    int64_t maxnbrs_per_atom_aligned_,
    int64_t angular_length_aligned_,
    int64_t ncenter_atoms_,
    Tensor coordinates_t_,
    Tensor species_t_) {
  aev_t = aev_t_;
  tensor_Rij = tensor_Rij_;
  tensor_radialRij = tensor_radialRij_;
  tensor_angularRij = tensor_angularRij_;
  total_natom_pairs = total_natom_pairs_;
  nRadialRij = nRadialRij_;
  nAngularRij = nAngularRij_;
  tensor_centralAtom = tensor_centralAtom_;
  tensor_numPairsPerCenterAtom = tensor_numPairsPerCenterAtom_;
  tensor_centerAtomStartIdx = tensor_centerAtomStartIdx_;
  maxnbrs_per_atom_aligned = maxnbrs_per_atom_aligned_;
  angular_length_aligned = angular_length_aligned_;
  ncenter_atoms = ncenter_atoms_;
  coordinates_t = coordinates_t_;
  species_t = species_t_;
}

void Result::release() {
  aev_t = Tensor();
  tensor_Rij = Tensor();
  tensor_radialRij = Tensor();
  tensor_angularRij = Tensor();
  tensor_centralAtom = Tensor();
  tensor_numPairsPerCenterAtom = Tensor();
  tensor_centerAtomStartIdx = Tensor();
  coordinates_t = Tensor();
  species_t = Tensor();
}

CuaevComputer::CuaevComputer(
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species) {
  aev_params.Rca = Rca;
  aev_params.Rcr = Rcr;
  aev_params.num_species = num_species;
  aev_params.radial_sublength = EtaR_t.size(0) * ShfR_t.size(0);
  aev_params.radial_length = aev_params.radial_sublength * num_species;
  aev_params.angular_sublength = EtaA_t.size(0) * Zeta_t.size(0) * ShfA_t.size(0) * ShfZ_t.size(0);
  aev_params.angular_length = aev_params.angular_sublength * (num_species * (num_species + 1) / 2);
  aev_params.EtaR_t = EtaR_t;
  aev_params.ShfR_t = ShfR_t;
  aev_params.EtaA_t = EtaA_t;
  aev_params.Zeta_t = Zeta_t;
  aev_params.ShfA_t = ShfA_t;
  aev_params.ShfZ_t = ShfZ_t;
}

Result CuaevComputer::forward(const Tensor& coordinates_t, const Tensor& species_t) {
  Result result = cuaev_forward(coordinates_t, species_t, aev_params);
  return result;
}

Tensor CuaevComputer::backward(const Tensor& grad_e_aev, const torch::intrusive_ptr<Result>& res_pt) {
  Tensor force = cuaev_backward(grad_e_aev, aev_params, res_pt);
  return force;
}

Tensor CuaevComputer::double_backward(const Tensor& grad_force, const torch::intrusive_ptr<Result>& res_pt) {
  Tensor grad_grad_aev = cuaev_double_backward(grad_force, aev_params, res_pt);
  return grad_grad_aev;
}

Tensor CuaevDoubleAutograd::forward(
    AutogradContext* ctx,
    Tensor grad_e_aev,
    AutogradContext* prectx,
    const torch::intrusive_ptr<Result>& res_pt) {
  torch::intrusive_ptr<CuaevComputer> cuaev_computer =
      prectx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  // torch::intrusive_ptr<Result> res_pt = prectx->saved_data["res_pt"].toCustomClass<Result>();;

  Tensor grad_coord = cuaev_computer->backward(grad_e_aev, res_pt);

  if (grad_e_aev.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->saved_data["res_pt"] = res_pt;
  } else {
    ctx->saved_data.erase("res_pt");
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0];
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  torch::intrusive_ptr<Result> res_pt = ctx->saved_data["res_pt"].toCustomClass<Result>();

  Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force, res_pt);
  ctx->saved_data.erase("res_pt");
  return {grad_grad_aev, torch::Tensor(), torch::Tensor()};
}

Tensor CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  at::AutoNonVariableTypeMode g;
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  if (coordinates_t.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->saved_data["res_pt"] = c10::make_intrusive<Result>(result);
  }
  return result.aev_t;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  torch::intrusive_ptr<Result> res_pt = ctx->saved_data["res_pt"].toCustomClass<Result>();
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], ctx, res_pt);
  return {grad_coord, Tensor(), Tensor()};
}

Tensor cuaev_only_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  return result.aev_t;
}

Tensor cuaev_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  return CuaevAutograd::apply(coordinates_t, species_t, cuaev_computer);
}

TORCH_LIBRARY(cuaev, m) {
  m.class_<Result>("Result").def(torch::init<
                                 Tensor,
                                 Tensor,
                                 Tensor,
                                 Tensor,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 Tensor,
                                 Tensor,
                                 Tensor,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 Tensor,
                                 Tensor>());
  m.class_<CuaevComputer>("CuaevComputer")
      .def(torch::init<double, double, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t>());
  m.def("run", cuaev_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("run", cuaev_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
