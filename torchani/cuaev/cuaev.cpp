#include <aev.h>
#include <bits/stdint-intn.h>
#include <torch/extension.h>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

AEVScalarParams::AEVScalarParams(const torch::IValue& aev_params_ivalue) {
  c10::intrusive_ptr<c10::ivalue::Tuple> aev_params_tuple_ptr = aev_params_ivalue.toTuple();
  auto aev_params_tuple = aev_params_tuple_ptr->elements();

  Rcr = static_cast<float>(aev_params_tuple[0].toDouble());
  Rca = static_cast<float>(aev_params_tuple[1].toDouble());
  radial_sublength = static_cast<int>(aev_params_tuple[2].toInt());
  radial_length = static_cast<int>(aev_params_tuple[3].toInt());
  angular_sublength = static_cast<int>(aev_params_tuple[4].toInt());
  angular_length = static_cast<int>(aev_params_tuple[5].toInt());
  num_species = static_cast<int>(aev_params_tuple[6].toInt());
}

struct Result1 : torch::CustomClassHolder {
  Tensor aev_t;
  Tensor tensor_Rij;
  Tensor tensor_radialRij;
  Tensor tensor_angularRij;
  int total_natom_pairs;
  int nRadialRij;
  int nAngularRij;
  Tensor tensor_centralAtom;
  Tensor tensor_numPairsPerCenterAtom;
  Tensor tensor_centerAtomStartIdx;
  int maxnbrs_per_atom_aligned;
  int angular_length_aligned;
  int ncenter_atoms;

  Result1(
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
      int64_t ncenter_atoms_)
      : torch::CustomClassHolder() {
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
  }
};

class CuaevDoubleAutograd : public torch::autograd::Function<CuaevDoubleAutograd> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor grad_e_aev, AutogradContext* prectx) {
    auto saved = prectx->get_saved_variables();
    auto coordinates_t = saved[0], species_t = saved[1];
    auto tensor_Rij = saved[2], tensor_radialRij = saved[3], tensor_angularRij = saved[4];
    auto EtaR_t = saved[5], ShfR_t = saved[6], EtaA_t = saved[7], Zeta_t = saved[8], ShfA_t = saved[9],
         ShfZ_t = saved[10];
    auto tensor_centralAtom = saved[11], tensor_numPairsPerCenterAtom = saved[12],
         tensor_centerAtomStartIdx = saved[13];
    AEVScalarParams aev_params(prectx->saved_data["aev_params"]);
    c10::List<int64_t> int_list = prectx->saved_data["int_list"].toIntList();
    int total_natom_pairs = int_list[0], nRadialRij = int_list[1], nAngularRij = int_list[2];
    int maxnbrs_per_atom_aligned = int_list[3], angular_length_aligned = int_list[4];
    int ncenter_atoms = int_list[5];
    // torch::intrusive_ptr<Result> res = prectx->saved_data["result"].toCustomClass<Result>();

    if (grad_e_aev.requires_grad()) {
      ctx->save_for_backward({coordinates_t,
                              species_t,
                              tensor_Rij,
                              tensor_radialRij,
                              tensor_angularRij,
                              EtaR_t,
                              ShfR_t,
                              EtaA_t,
                              Zeta_t,
                              ShfA_t,
                              ShfZ_t,
                              tensor_centralAtom,
                              tensor_numPairsPerCenterAtom,
                              tensor_centerAtomStartIdx});
      ctx->saved_data["aev_params"] = aev_params;
      ctx->saved_data["int_list"] = c10::List<int64_t>{
          total_natom_pairs, nRadialRij, nAngularRij, maxnbrs_per_atom_aligned, angular_length_aligned, ncenter_atoms};
    }

    Tensor grad_coord = cuaev_backward(
        grad_e_aev,
        coordinates_t,
        species_t,
        aev_params,
        EtaR_t,
        ShfR_t,
        EtaA_t,
        Zeta_t,
        ShfA_t,
        ShfZ_t,
        tensor_Rij,
        total_natom_pairs,
        tensor_radialRij,
        nRadialRij,
        tensor_angularRij,
        nAngularRij,
        tensor_centralAtom,
        tensor_numPairsPerCenterAtom,
        tensor_centerAtomStartIdx,
        maxnbrs_per_atom_aligned,
        angular_length_aligned,
        ncenter_atoms);

    return grad_coord;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_force = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    auto coordinates_t = saved[0], species_t = saved[1];
    auto tensor_Rij = saved[2], tensor_radialRij = saved[3], tensor_angularRij = saved[4];
    auto EtaR_t = saved[5], ShfR_t = saved[6], EtaA_t = saved[7], Zeta_t = saved[8], ShfA_t = saved[9],
         ShfZ_t = saved[10];
    auto tensor_centralAtom = saved[11], tensor_numPairsPerCenterAtom = saved[12],
         tensor_centerAtomStartIdx = saved[13];
    AEVScalarParams aev_params(ctx->saved_data["aev_params"]);
    c10::List<int64_t> int_list = ctx->saved_data["int_list"].toIntList();
    int total_natom_pairs = int_list[0], nRadialRij = int_list[1], nAngularRij = int_list[2];
    int maxnbrs_per_atom_aligned = int_list[3], angular_length_aligned = int_list[4];
    int ncenter_atoms = int_list[5];

    Tensor grad_grad_aev = cuaev_double_backward(
        grad_force,
        coordinates_t,
        species_t,
        aev_params,
        EtaR_t,
        ShfR_t,
        EtaA_t,
        Zeta_t,
        ShfA_t,
        ShfZ_t,
        tensor_Rij,
        total_natom_pairs,
        tensor_radialRij,
        nRadialRij,
        tensor_angularRij,
        nAngularRij,
        tensor_centralAtom,
        tensor_numPairsPerCenterAtom,
        tensor_centerAtomStartIdx,
        maxnbrs_per_atom_aligned,
        angular_length_aligned,
        ncenter_atoms);

    return {grad_grad_aev, torch::Tensor()};
  }
};

#define AEV_INPUT                                                                                                   \
  const Tensor &coordinates_t, const Tensor &species_t, double Rcr_, double Rca_, const Tensor &EtaR_t,             \
      const Tensor &ShfR_t, const Tensor &EtaA_t, const Tensor &Zeta_t, const Tensor &ShfA_t, const Tensor &ShfZ_t, \
      int64_t num_species_

Tensor cuaev_cuda(AEV_INPUT) {
  Result res =
      cuaev_forward(coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
  return res.aev_t;
}

class CuaevAutograd : public torch::autograd::Function<CuaevAutograd> {
 public:
  static Tensor forward(AutogradContext* ctx, AEV_INPUT) {
    at::AutoNonVariableTypeMode g;
    Result res = cuaev_forward(
        coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
    if (coordinates_t.requires_grad()) {
      ctx->save_for_backward({coordinates_t,
                              species_t,
                              res.tensor_Rij,
                              res.tensor_radialRij,
                              res.tensor_angularRij,
                              EtaR_t,
                              ShfR_t,
                              EtaA_t,
                              Zeta_t,
                              ShfA_t,
                              ShfZ_t,
                              res.tensor_centralAtom,
                              res.tensor_numPairsPerCenterAtom,
                              res.tensor_centerAtomStartIdx});
      ctx->saved_data["aev_params"] = res.aev_params;
      //   ctx->saved_data["result"] = c10::make_intrusive<Result>(res);
      // ctx->saved_data["result"] = (torch::intrusive_ptr<Result>) res;
      ctx->saved_data["int_list"] = c10::List<int64_t>{res.total_natom_pairs,
                                                       res.nRadialRij,
                                                       res.nAngularRij,
                                                       res.maxnbrs_per_atom_aligned,
                                                       res.angular_length_aligned,
                                                       res.ncenter_atoms};
    }
    return res.aev_t;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], ctx);

    return {
        grad_coord, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

Tensor cuaev_autograd(AEV_INPUT) {
  return CuaevAutograd::apply(
      coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
}

// struct CuaevComputer : torch::CustomClassHolder {

// };

TORCH_LIBRARY(cuaev, m) {
  m.class_<Result1>("Result1").def(torch::init<
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
                                   int64_t>());
  m.def("cuComputeAEV", cuaev_cuda);
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("cuComputeAEV", cuaev_cuda);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("cuComputeAEV", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
