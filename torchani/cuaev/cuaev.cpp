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

Tensor CuaevComputer::forward(const Tensor& coordinates_t, const Tensor& species_t) {
  cuaev_forward(coordinates_t, species_t, aev_params, result);
  return result.aev_t;
}

Tensor CuaevComputer::backward(const Tensor& grad_e_aev) {
  Tensor force = cuaev_backward(grad_e_aev, aev_params, result);
  return force;
}

Tensor CuaevComputer::double_backward(const Tensor& grad_force) {
  Tensor grad_grad_aev = cuaev_double_backward(grad_force, aev_params, result);
  return grad_grad_aev;
}

Tensor CuaevDoubleAutograd::forward(AutogradContext* ctx, Tensor grad_e_aev, AutogradContext* prectx) {
  torch::intrusive_ptr<CuaevComputer> cuaev_computer =
      prectx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();

  Tensor grad_coord = cuaev_computer->backward(grad_e_aev);

  if (grad_e_aev.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
  } else {
    // ctx->saved_data.erase("cuaev_computer");
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0];
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();

  Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force);

  return {grad_grad_aev, torch::Tensor()};
}

Tensor CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  at::AutoNonVariableTypeMode g;
  Tensor aev_t = cuaev_computer->forward(coordinates_t, species_t);
  if (coordinates_t.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
  }
  return aev_t;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], ctx);
  return {grad_coord, Tensor(), Tensor()};
}

Tensor cuaev_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  return CuaevAutograd::apply(coordinates_t, species_t, cuaev_computer);
}

TORCH_LIBRARY(cuaev, m) {
  m.class_<CuaevComputer>("CuaevComputer")
      .def(torch::init<double, double, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t>());
  m.def("run", cuaev_autograd);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
