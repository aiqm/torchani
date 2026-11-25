#include <torch/all.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

#include "aev.h"

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

// The forward ops return the aev, followed by the tensors that the backward needs and that
// are not inputs of the forward: {sizes, atomI, startIdxJ, radial nbrlist, angular nbrlist}, where
// sizes is {nI, radial nJ, radial maxNumJPerI, angular nJ, angular maxNumJPerI} and each nbrlist is
// {atomJ, numJPerI, distJ, deltaJ, pairIdxJ}.
// Op outputs can't be undefined, so undefined tensors (e.g. pairIdxJ if the half nbrlist is not
// used) are returned as empty tensors.
static Tensor defined_or_empty(const Tensor& t, const torch::TensorOptions& options) {
  return t.defined() ? t : torch::empty({0}, options);
}

static void append_neighbor_list(tensor_list& tensors, const NeighborList& nbr, const torch::TensorOptions& options) {
  torch::TensorOptions int_options = options.dtype(torch::kInt32);
  tensors.push_back(defined_or_empty(nbr.atomJ_t, int_options));
  tensors.push_back(defined_or_empty(nbr.numJPerI_t, int_options));
  tensors.push_back(defined_or_empty(nbr.distJ_t, options));
  tensors.push_back(defined_or_empty(nbr.deltaJ_t, options));
  tensors.push_back(defined_or_empty(nbr.pairIdxJ_t, int_options));
}

static tensor_list forward_outputs(const Result& result) {
  torch::TensorOptions options = result.aev_t.options();
  torch::TensorOptions int_options = options.dtype(torch::kInt32);
  std::vector<int64_t> sizes = {
      result.nI,
      result.radialNbr.nJ,
      result.radialNbr.maxNumJPerI,
      result.angularNbr.nJ,
      result.angularNbr.maxNumJPerI};
  tensor_list outputs = {
      result.aev_t,
      torch::tensor(sizes, torch::kInt64),
      defined_or_empty(result.atomI_t, int_options),
      defined_or_empty(result.startIdxJ_t, int_options)};
  append_neighbor_list(outputs, result.radialNbr, options);
  append_neighbor_list(outputs, result.angularNbr, options);
  return outputs;
}

// pairIdxJ is only computed for the half nbrlist. Op outputs can't be undefined, so otherwise it
// is returned as an empty tensor, and here an empty pairIdxJ is read back as undefined. The half
// nbrlist with no pairs also gives an empty pairIdxJ, which is fine only because then nI == 0 and
// the backward kernels don't run.
static NeighborList saved_neighbor_list(at::TensorList saved, int64_t offset, int64_t nJ, int64_t maxNumJPerI) {
  Tensor pairIdxJ = saved[offset + 4].numel() > 0 ? saved[offset + 4] : Tensor();
  return NeighborList(
      nJ, maxNumJPerI, saved[offset], saved[offset + 1], saved[offset + 2], saved[offset + 3], pairIdxJ);
}

// saved holds the outputs of the forward ops, except for the aev
static Result saved_to_result(at::TensorList saved, const Tensor& grad_target, const Tensor& species) {
  auto sizes = saved[0].accessor<int64_t, 1>();
  return Result(
      Tensor(),
      saved[1],
      saved[2],
      sizes[0],
      grad_target,
      species,
      saved_neighbor_list(saved, 3, sizes[1], sizes[2]),
      saved_neighbor_list(saved, 8, sizes[3], sizes[4]));
}

// Only the aev is differentiable. Grads are not materialized, otherwise autograd allocates zeros
// for the grads of all the other outputs before calling the backward
static void setup_non_differentiable_outputs(AutogradContext* ctx, const tensor_list& outputs) {
  ctx->mark_non_differentiable(tensor_list(outputs.begin() + 1, outputs.end()));
  ctx->set_materialize_grads(false);
}

static Tensor pop_back_tensor(tensor_list& tensors) {
  Tensor t = tensors.back();
  tensors.pop_back();
  return t;
}

// CUDA kernels
tensor_list run_cuda(
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
  return forward_outputs(aev_forward(coordinates_t, species_t, aev_params));
}

tensor_list run_with_half_nbrlist_cuda(
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
  return forward_outputs(aev_forward_with_half_nbrlist(species_t, atomIJ_t, deltaJ_t, distJ_t, aev_params));
}

tensor_list run_with_full_nbrlist_cuda(
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
  return forward_outputs(
      aev_forward_with_full_nbrlist(coordinates_t, species_t, atomI_t, atomJ_t, numJPerI_t, aev_params));
}

Tensor run_backward(
    const Tensor& grad_aev,
    const Tensor& grad_target,
    const Tensor& species,
    at::TensorList saved,
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
  return aev_backward(grad_aev, aev_params, saved_to_result(saved, grad_target, species));
}

Tensor run_double_backward(
    const Tensor& grad_force,
    const Tensor& grad_target,
    const Tensor& species,
    at::TensorList saved,
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
  return aev_double_backward(grad_force, aev_params, saved_to_result(saved, grad_target, species));
}

// The autograd functions call the ops through the dispatcher, so that tracing
// (e.g. torch.compile) sees the ops instead of running the CUDA kernels
static tensor_list call_run_op(const Tensor& coordinates_t, const Tensor& species_t, const AEVScalarParams& p) {
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("cuaev::run", "").typed<decltype(run_cuda)>();
  return op.call(
      coordinates_t,
      species_t,
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

static tensor_list call_run_with_half_nbrlist_op(
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    const AEVScalarParams& p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("cuaev::run_with_half_nbrlist", "")
                       .typed<decltype(run_with_half_nbrlist_cuda)>();
  return op.call(
      species_t,
      atomIJ_t,
      deltaJ_t,
      distJ_t,
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

static tensor_list call_run_with_full_nbrlist_op(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    const AEVScalarParams& p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("cuaev::run_with_full_nbrlist", "")
                       .typed<decltype(run_with_full_nbrlist_cuda)>();
  return op.call(
      coordinates_t,
      species_t,
      atomI_t,
      atomJ_t,
      numJPerI_t,
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

static Tensor call_backward_op(
    const Tensor& grad_aev,
    const Tensor& grad_target,
    const Tensor& species,
    const tensor_list& saved,
    const AEVScalarParams& p) {
  static auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("cuaev::_backward", "").typed<decltype(run_backward)>();
  return op.call(
      grad_aev,
      grad_target,
      species,
      saved,
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

static Tensor call_double_backward_op(
    const Tensor& grad_force,
    const Tensor& grad_target,
    const Tensor& species,
    const tensor_list& saved,
    const AEVScalarParams& p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("cuaev::_double_backward", "")
                       .typed<decltype(run_double_backward)>();
  return op.call(
      grad_force,
      grad_target,
      species,
      saved,
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

Tensor CuaevDoubleAutograd::forward(
    AutogradContext* ctx,
    Tensor grad_e_aev,
    Tensor grad_target,
    Tensor species,
    const AEVScalarParams& aev_params,
    tensor_list saved) {
  at::AutoDispatchBelowADInplaceOrView guard;
  grad_e_aev = grad_e_aev.contiguous();
  Tensor grad_coord = call_backward_op(grad_e_aev, grad_target, species, saved, aev_params);

  if (grad_e_aev.requires_grad() || grad_target.requires_grad()) {
    save_aev_params(ctx, aev_params);
    saved.push_back(grad_target);
    saved.push_back(species);
    ctx->save_for_backward(saved);
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0].contiguous();
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list saved = ctx->get_saved_variables();
  Tensor species = pop_back_tensor(saved);
  Tensor grad_target = pop_back_tensor(saved);

  Tensor grad_grad_aev;
  if (ctx->needs_input_grad(0)) {
    grad_grad_aev = call_double_backward_op(grad_force, grad_target, species, saved, aev_params);
  }
  if (ctx->needs_input_grad(1)) {
    TORCH_WARN_ONCE(
        "cuAEV doesn't support second derivatives wrt the coordinates. Gradients that depend on them "
        "(e.g. hessians) are missing the term that depends on the second derivatives of the AEV");
  }
  return {grad_grad_aev, Tensor(), Tensor(), Tensor(), Tensor()};
}

tensor_list CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  tensor_list outputs = call_run_op(coordinates_t, species_t, aev_params);
  setup_non_differentiable_outputs(ctx, outputs);
  if (coordinates_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    tensor_list saved(outputs.begin() + 1, outputs.end());
    saved.push_back(coordinates_t);
    saved.push_back(species_t);
    ctx->save_for_backward(saved);
  }
  return outputs;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list saved = ctx->get_saved_variables();
  Tensor species = pop_back_tensor(saved);
  Tensor coordinates = pop_back_tensor(saved);
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], coordinates, species, aev_params, saved);
  return {grad_coord, Tensor(), Tensor()};
}

tensor_list CuaevWithHalfNbrlistAutograd::forward(
    AutogradContext* ctx,
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  tensor_list outputs = call_run_with_half_nbrlist_op(species_t, atomIJ_t, deltaJ_t, distJ_t, aev_params);
  setup_non_differentiable_outputs(ctx, outputs);
  if (deltaJ_t.requires_grad() || distJ_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    // deltaJ is saved so that the backward can build a view of it tracked by autograd
    tensor_list saved(outputs.begin() + 1, outputs.end());
    saved.push_back(deltaJ_t);
    saved.push_back(species_t);
    ctx->save_for_backward(saved);
  }
  return outputs;
}

// The full gradient wrt the pair vectors (including the dependence of the distances on them)
// is returned for deltaJ, so no gradient is returned for distJ
tensor_list CuaevWithHalfNbrlistAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list saved = ctx->get_saved_variables();
  Tensor species = pop_back_tensor(saved);
  Tensor deltaJ = pop_back_tensor(saved);
  Tensor grad_deltaJ =
      CuaevDoubleAutograd::apply(grad_outputs[0], deltaJ.view({1, -1, 3}), species, aev_params, saved).view({-1, 3});
  return {Tensor(), Tensor(), grad_deltaJ, Tensor(), Tensor()};
}

tensor_list CuaevWithFullNbrlistAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    const AEVScalarParams& aev_params) {
  at::AutoDispatchBelowADInplaceOrView guard;
  tensor_list outputs =
      call_run_with_full_nbrlist_op(coordinates_t, species_t, atomI_t, atomJ_t, numJPerI_t, aev_params);
  setup_non_differentiable_outputs(ctx, outputs);
  if (coordinates_t.requires_grad()) {
    save_aev_params(ctx, aev_params);
    tensor_list saved(outputs.begin() + 1, outputs.end());
    saved.push_back(coordinates_t);
    saved.push_back(species_t);
    ctx->save_for_backward(saved);
  }
  return outputs;
}

tensor_list CuaevWithFullNbrlistAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  AEVScalarParams aev_params = load_aev_params(ctx);
  tensor_list saved = ctx->get_saved_variables();
  Tensor species = pop_back_tensor(saved);
  Tensor coordinates = pop_back_tensor(saved);
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], coordinates, species, aev_params, saved);
  return {grad_coord, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
}

tensor_list run_autograd(
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

tensor_list run_with_half_nbrlist_autograd(
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

tensor_list run_with_full_nbrlist_autograd(
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

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("run", run_cuda);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_cuda);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_cuda);
  m.impl("_backward", run_backward);
  m.impl("_double_backward", run_double_backward);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", run_autograd);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_autograd);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_autograd);
  // The backward ops are not differentiable, this errors if a gradient flows through them
  m.impl("_backward", torch::autograd::autogradNotImplementedFallback());
  m.impl("_double_backward", torch::autograd::autogradNotImplementedFallback());
}
