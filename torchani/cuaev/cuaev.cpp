#include <aev.h>
#include <torch/extension.h>
#include <iostream>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class CuaevDoubleAutograd : public torch::autograd::Function<CuaevDoubleAutograd> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor grad_e_aev, AutogradContext* prectx) {
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

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_force = grad_outputs[0];
    torch::intrusive_ptr<CuaevComputer> cuaev_computer =
        ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();

    Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force);

    return {grad_grad_aev, torch::Tensor()};
  }
};

class CuaevAutograd : public torch::autograd::Function<CuaevAutograd> {
 public:
  static Tensor forward(
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

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], ctx);
    return {grad_coord, Tensor(), Tensor()};
  }
};

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
  m.impl("cuComputeAEV", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
