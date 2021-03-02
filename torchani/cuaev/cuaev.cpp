#include <aev.h>
#include <torch/extension.h>
#include <iostream>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class CuaevDoubleAutograd : public torch::autograd::Function<CuaevDoubleAutograd> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor grad_e_aev, AutogradContext* prectx) {
    torch::intrusive_ptr<Result> res = prectx->saved_data["result"].toCustomClass<Result>();
    torch::intrusive_ptr<CuaevComputer> cuaev_computer =
        prectx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();

    Tensor grad_coord = cuaev_computer->backward(grad_e_aev, res);

    if (grad_e_aev.requires_grad()) {
      ctx->saved_data["result"] = res;
      ctx->saved_data["cuaev_computer"] = cuaev_computer;
    } else {
      // ctx->saved_data.erase("result");
      res->release();
    }

    return grad_coord;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_force = grad_outputs[0];
    torch::intrusive_ptr<Result> res = ctx->saved_data["result"].toCustomClass<Result>();
    torch::intrusive_ptr<CuaevComputer> cuaev_computer =
        ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();

    Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force, res);

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
    Result res = cuaev_computer->forward(coordinates_t, species_t);
    if (coordinates_t.requires_grad()) {
      ctx->saved_data["result"] = c10::make_intrusive<Result>(res);
      ctx->saved_data["cuaev_computer"] = cuaev_computer;
    }
    return res.aev_t;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], ctx);
    // ctx->saved_data.erase("result");
    // torch::intrusive_ptr<Result> res = ctx->saved_data["result"].toCustomClass<Result>();
    // res->release();
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
  // .def("forward", &CuaevComputer::forward)
  // .def("backward", &CuaevComputer::backward)
  // .def("double_backward", &CuaevComputer::double_backward);
  m.def("run", cuaev_autograd);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("cuComputeAEV", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
