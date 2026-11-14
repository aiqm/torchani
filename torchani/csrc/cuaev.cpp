#include <torch/extension.h>

#include "aev.h"
#include "aev_cpu.h"

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

AEVScalarParams::AEVScalarParams(
    double Rcr,
    double Rca,
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
    Tensor grad_target_t,
    Tensor species_t,
    NeighborList radialNbr,
    NeighborList angularNbr)
    : aev_t(aev_t),
      atomI_t(atomI_t),
      startIdxJ_t(startIdxJ_t),
      nI(nI),
      grad_target_t(grad_target_t),
      species_t(species_t),
      radialNbr(radialNbr),
      angularNbr(angularNbr) {}

Result::Result(tensor_list tensors)
    : aev_t(tensors[0]), // aev_t will be a undefined tensor
      atomI_t(tensors[1]),
      startIdxJ_t(tensors[2]),
      nI(tensors[3].item<int>()),
      grad_target_t(tensors[4]),
      species_t(tensors[5]),
      radialNbr(
          tensors[6].item<int>(),
          tensors[7].item<int>(),
          tensors[8],
          tensors[9],
          tensors[10],
          tensors[11],
          tensors[12]),
      angularNbr(
          tensors[13].item<int>(),
          tensors[14].item<int>(),
          tensors[15],
          tensors[16],
          tensors[17],
          tensors[18],
          tensors[19]) {}

Result::Result()
    : aev_t(Tensor()),
      atomI_t(Tensor()),
      startIdxJ_t(Tensor()),
      nI(0),
      grad_target_t(Tensor()),
      species_t(Tensor()),
      radialNbr(NeighborList()),
      angularNbr(NeighborList()) {}

Result::Result(Tensor grad_target_t, Tensor species_t)
    : aev_t(Tensor()),
      atomI_t(Tensor()),
      startIdxJ_t(Tensor()),
      nI(0),
      grad_target_t(grad_target_t),
      species_t(species_t),
      radialNbr(NeighborList()),
      angularNbr(NeighborList()) {}

static void save_aev_params(AutogradContext* ctx, const AEVScalarParams& aev_params) {
  ctx->saved_data["Rcr"] = aev_params.Rcr;
  ctx->saved_data["Rca"] = aev_params.Rca;
  ctx->saved_data["EtaR_t"] = aev_params.EtaR_t;
  ctx->saved_data["ShfR_t"] = aev_params.ShfR_t;
  ctx->saved_data["EtaA_t"] = aev_params.EtaA_t;
  ctx->saved_data["Zeta_t"] = aev_params.Zeta_t;
  ctx->saved_data["ShfA_t"] = aev_params.ShfA_t;
  ctx->saved_data["ShfZ_t"] = aev_params.ShfZ_t;
  ctx->saved_data["num_species"] = aev_params.num_species;
  ctx->saved_data["use_cos_cutoff"] = aev_params.use_cos_cutoff;
}

static AEVScalarParams load_aev_params(AutogradContext* ctx) {
  return AEVScalarParams(
      ctx->saved_data["Rcr"].toDouble(),
      ctx->saved_data["Rca"].toDouble(),
      ctx->saved_data["EtaR_t"].toTensor(),
      ctx->saved_data["ShfR_t"].toTensor(),
      ctx->saved_data["EtaA_t"].toTensor(),
      ctx->saved_data["Zeta_t"].toTensor(),
      ctx->saved_data["ShfA_t"].toTensor(),
      ctx->saved_data["ShfZ_t"].toTensor(),
      ctx->saved_data["num_species"].toInt(),
      ctx->saved_data["use_cos_cutoff"].toBool());
}

// Product of the Hessian of (aev * grad_aev).sum() wrt grad_target with grad_grad_target.
// The CUDA kernels don't compute this term, so the AEV is recomputed with the ATen impl.
static Tensor aev_hessian_product(
    CuaevOp op,
    const Tensor& grad_target,
    const Tensor& species,
    const tensor_list& aten_inputs,
    const AEVScalarParams& p,
    const Tensor& grad_aev,
    const Tensor& grad_grad_target) {
  torch::AutoGradMode enable_grad(true);
  Tensor target = grad_target.detach().requires_grad_(true);
  Tensor aev;
  if (op == CuaevOp::Fused) {
    aev = run_aten(
        target,
        species,
        p.Rcr,
        p.Rca,
        p.EtaR_t,
        p.ShfR_t,
        p.EtaA_t,
        p.Zeta_t,
        p.ShfA_t,
        p.ShfZ_t,
        p.num_species,
        p.use_cos_cutoff);
  } else if (op == CuaevOp::HalfNbrlist) {
    Tensor diff_vectors = target.view({-1, 3});
    aev = run_with_half_nbrlist_aten(
        species,
        aten_inputs[0],
        diff_vectors,
        diff_vectors.norm(2, -1),
        p.Rcr,
        p.Rca,
        p.EtaR_t,
        p.ShfR_t,
        p.EtaA_t,
        p.Zeta_t,
        p.ShfA_t,
        p.ShfZ_t,
        p.num_species,
        p.use_cos_cutoff);
  } else {
    aev = run_with_full_nbrlist_aten(
        target,
        species,
        aten_inputs[0],
        aten_inputs[1],
        aten_inputs[2],
        p.Rcr,
        p.Rca,
        p.EtaR_t,
        p.ShfR_t,
        p.EtaA_t,
        p.Zeta_t,
        p.ShfA_t,
        p.ShfZ_t,
        p.num_species,
        p.use_cos_cutoff);
  }
  Tensor energy = (aev * grad_aev.detach()).sum();
  Tensor grad = torch::autograd::grad({energy}, {target}, {}, true, true, true)[0];
  if (!grad.defined() || !grad.requires_grad()) {
    return torch::zeros_like(grad_target);
  }
  Tensor product = torch::autograd::grad({grad}, {target}, {grad_grad_target.detach()}, false, false, true)[0];
  if (!product.defined()) {
    return torch::zeros_like(grad_target);
  }
  return product;
}

// grad_target is also part of result_tensors, it is passed separately so that autograd tracks it
Tensor CuaevDoubleAutograd::forward(
    AutogradContext* ctx,
    Tensor grad_e_aev,
    Tensor grad_target,
    const AEVScalarParams& aev_params,
    tensor_list result_tensors,
    tensor_list aten_inputs,
    CuaevOp op) {
  Tensor grad_coord = aev_backward(grad_e_aev, aev_params, result_tensors);

  if (grad_e_aev.requires_grad() || grad_target.requires_grad()) {
    save_aev_params(ctx, aev_params);
    ctx->saved_data["aten_inputs"] = aten_inputs;
    ctx->saved_data["op"] = static_cast<int64_t>(op);
    result_tensors.push_back(grad_e_aev);
    ctx->save_for_backward(result_tensors);
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0];
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor grad_e_aev = result_tensors.back();
  result_tensors.pop_back();

  Tensor grad_grad_aev;
  if (ctx->needs_input_grad(0)) {
    grad_grad_aev = aev_double_backward(grad_force, aev_params, result_tensors);
  }
  Tensor grad_grad_target;
  if (ctx->needs_input_grad(1)) {
    grad_grad_target = aev_hessian_product(
        static_cast<CuaevOp>(ctx->saved_data["op"].toInt()),
        result_tensors[4],
        result_tensors[5],
        ctx->saved_data["aten_inputs"].toTensorVector(),
        aev_params,
        grad_e_aev,
        grad_force);
  }
  return {grad_grad_aev, grad_grad_target, Tensor(), Tensor(), Tensor(), Tensor()};
}

Tensor CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  Result result = aev_forward(coordinates_t, species_t, aev_params);
  if (coordinates_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    ctx->save_for_backward(result);
  }
  return result.aev_t;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor grad_coord = CuaevDoubleAutograd::apply(
      grad_outputs[0], result_tensors[4], aev_params, result_tensors, tensor_list{}, CuaevOp::Fused);
  return {grad_coord, Tensor(), Tensor()};
}

Tensor CuaevWithHalfNbrlistAutograd::forward(
    AutogradContext* ctx,
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  Result result = aev_forward_with_half_nbrlist(species_t, atomIJ_t, deltaJ_t, distJ_t, aev_params);
  if (deltaJ_t.requires_grad() || distJ_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    ctx->saved_data["atomIJ"] = atomIJ_t;
    // deltaJ is saved so that the backward can build a view of it tracked by autograd
    tensor_list result_tensors = result;
    result_tensors.push_back(deltaJ_t);
    ctx->save_for_backward(result_tensors);
  }
  return result.aev_t;
}

// The full gradient wrt the pair vectors (including the dependence of the distances on them)
// is returned for deltaJ, so no gradient is returned for distJ
tensor_list CuaevWithHalfNbrlistAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor deltaJ = result_tensors.back();
  result_tensors.pop_back();
  result_tensors[4] = deltaJ.view({1, -1, 3});
  tensor_list aten_inputs = {ctx->saved_data["atomIJ"].toTensor()};
  Tensor grad_deltaJ =
      CuaevDoubleAutograd::apply(
          grad_outputs[0], result_tensors[4], aev_params, result_tensors, aten_inputs, CuaevOp::HalfNbrlist)
          .view({-1, 3});
  return {Tensor(), Tensor(), grad_deltaJ, Tensor(), Tensor()};
}

Tensor CuaevWithFullNbrlistAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  Result result = aev_forward_with_full_nbrlist(coordinates_t, species_t, atomI_t, atomJ_t, numJPerI_t, aev_params);
  if (coordinates_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    ctx->saved_data["aten_inputs"] = tensor_list{atomI_t, atomJ_t, numJPerI_t};
    ctx->save_for_backward(result);
  }
  return result.aev_t;
}

tensor_list CuaevWithFullNbrlistAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor grad_coord = CuaevDoubleAutograd::apply(
      grad_outputs[0],
      result_tensors[4],
      aev_params,
      result_tensors,
      ctx->saved_data["aten_inputs"].toTensorVector(),
      CuaevOp::FullNbrlist);
  return {grad_coord, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
}

Tensor run_only_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  Result result = aev_forward(coordinates_t, species_t, aev_params);
  return result.aev_t;
}

Tensor run_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  return CuaevAutograd::apply(coordinates_t, species_t, aev_params);
}

Tensor run_with_half_nbrlist_only_forward(
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  Result result = aev_forward_with_half_nbrlist(species_t, atomIJ_t, deltaJ_t, distJ_t, aev_params);
  return result.aev_t;
}

Tensor run_with_half_nbrlist_autograd(
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  return CuaevWithHalfNbrlistAutograd::apply(species_t, atomIJ_t, deltaJ_t, distJ_t, aev_params);
}

Tensor run_with_full_nbrlist_only_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  Result result = aev_forward_with_full_nbrlist(coordinates_t, species_t, atomI_t, atomJ_t, numJPerI_t, aev_params);
  return result.aev_t;
}

Tensor run_with_full_nbrlist_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species,
    bool use_cos_cutoff) {
  AEVScalarParams aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species, use_cos_cutoff);
  return CuaevWithFullNbrlistAutograd::apply(coordinates_t, species_t, atomI_t, atomJ_t, numJPerI_t, aev_params);
}

// Explicit schemas also keep the argument names in python
TORCH_LIBRARY(cuaev, m) {
  m.def(
      "run(Tensor coordinates, Tensor species, float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, "
      "Tensor Zeta, Tensor ShfA, Tensor ShfZ, int num_species, bool use_cos_cutoff) -> Tensor");
  m.def(
      "run_with_half_nbrlist(Tensor species, Tensor atomIJ, Tensor deltaJ, Tensor distJ, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor");
  m.def(
      "run_with_full_nbrlist(Tensor coordinates, Tensor species, Tensor atomI, Tensor atomJ, Tensor numJPerI, "
      "float Rcr, float Rca, Tensor EtaR, Tensor ShfR, Tensor EtaA, Tensor Zeta, Tensor ShfA, Tensor ShfZ, "
      "int num_species, bool use_cos_cutoff) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("run", run_only_forward);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_only_forward);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", run_autograd);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_autograd);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_autograd);
}
