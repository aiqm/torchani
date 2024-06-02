#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

// [Computation graph for forward, backward, and double backward]
//
// backward
// force = (dE / daev) * (daev / dcoord) = g * (daev / dcoord)
//
// double backward (to do force training, the term needed is)
// dloss / dg = (dloss / dforce) * (dforce / dg) = (dloss / dforce) * (daev / dcoord)
//
//
// [Forward]
//            out               ^
//             |                ^
//            ...               ^
//             |                ^
//        e n e r g y           ^
//           |     \            ^
//          aev     \           ^
//        /   |      \          ^
//  radial  angular  params     ^
//    /    /  |                 ^
// dist---^  /                  ^
//    \     /                   ^
//     coord                    ^
//
// Functional relationship:
// coord <-- input
// dist(coord)
// radial(dist)
// angular(dist, coord)
// aev = concatenate(radial, angular)
// energy(aev, params)
// out(energy, ....) <-- output
//
//
// [Backward]
//                   dout                     v
//                    |                       v
//                   ...                      v
//                    |                       v
//       aev params denergy  aev params       v
//         \   |   /      \   |   /           v
//          d a e v        dparams            v
//          /      \____                      v
// dist dradial         \                     v
//   \    /              \                    v
//   ddist dist coord   dangular dist coord   v
//      \   /    /           \    |    /      v
//       \_/____/             \___|___/       v
//        |    __________________/            v
//        |   /                               v
//      dcoord                                v
//        |                                   v
//       ...                                  v
//        |                                   v
//       out2                                 v
//
// Functional relationship:
// dout <-- input
// denergy(dout)
// dparams(denergy, aev, params)  <-- output
// daev(denergy, aev, params)
// dradial = slice(daev)
// dangular = slice(daev)
// ddist = radial_backward(dradial, dist) + angular_backward_dist(dangular, ...)
//       = radial_backward(dradial, dist) + 0 (all contributions route to dcoord)
//       = radial_backward(dradial, dist)
// dcoord = dist_backward(ddist, coord, dist) + angular_backward_coord(dangular, coord, dist)
// out2(dcoord, ...)  <-- output
//
//
// [Double backward w.r.t params (i.e. force training)]
// Note: only a very limited subset of double backward is implemented
// currently it can only do force training, there is no hessian support
// not implemented terms are marked by $s
//      $$$ [dparams] $$$$                     ^
//         \_  |    __/                        ^
//           [ddaev]                           ^
//          /      \_____                      ^
// $$$$ [ddradial]        \                    ^
//   \    /                \                   ^
//  [dddist] $$$$ $$$$ [ddangular] $$$$  $$$$  ^
//       \   /    /           \      |    /    ^
//        \_/____/             \_____|___/     ^
//         |    _____________________/         ^
//         |   /                               ^
//      [ddcoord]                              ^
//         |                                   ^
//        ...                                  ^
//         |                                   ^
//       [dout2]                               ^
//
// Functional relationship:
// dout2 <-- input
// ddcoord(dout2, ...)
// dddist = dist_doublebackward(ddcoord, coord, dist)
// ddradial = radial_doublebackward(dddist, dist)
// ddangular = angular_doublebackward(ddcord, coord, dist)
// ddaev = concatenate(ddradial, ddangular)
// dparams(ddaev, ...) <-- output

struct alignas(2 * sizeof(int)) AtomI {
  int midx;
  int i;
};

struct NeighborList {
  int nJ;
  int maxNumJPerI;

  Tensor atomJ_t; // only j index
  Tensor numJPerI_t;
  Tensor distJ_t;
  Tensor deltaJ_t;

  NeighborList() = default;
  NeighborList(int nJ, int maxNumJPerI, Tensor atomJ_t, Tensor numJPerI_t, Tensor distJ_t, Tensor deltaJ_t)
      : nJ(nJ),
        maxNumJPerI(maxNumJPerI),
        atomJ_t(atomJ_t),
        numJPerI_t(numJPerI_t),
        distJ_t(distJ_t),
        deltaJ_t(deltaJ_t) {}
};

struct AEVScalarParams {
  double Rcr;
  double Rca;
  int radial_sublength;
  int radial_length;
  int angular_sublength;
  int angular_length;
  int num_species;
  Tensor EtaR_t;
  Tensor ShfR_t;
  Tensor EtaA_t;
  Tensor Zeta_t;
  Tensor ShfA_t;
  Tensor ShfZ_t;
  bool use_cos_cutoff;

  AEVScalarParams(
      double Rcr,
      double Rca,
      Tensor EtaR_t,
      Tensor ShfR_t,
      Tensor EtaA_t,
      Tensor Zeta_t,
      Tensor ShfA_t,
      Tensor ShfZ_t,
      int num_species,
      bool use_cos_cutoff);
};

struct Result {
  Tensor aev_t;
  Tensor atomI_t;
  Tensor startIdxJ_t;
  int nI;
  Tensor coordinates_t;
  Tensor species_t;
  NeighborList radialNbr;
  NeighborList angularNbr;

  Result(
      Tensor aev_t,
      Tensor atomI_t,
      Tensor startIdxJ_t,
      int64_t nI,
      Tensor coordinates_t,
      Tensor species_t,
      NeighborList radialNbr,
      NeighborList angularNbr);
  Result(tensor_list tensors);
  Result(Tensor coordinates_t, Tensor species_t);
  Result();
  operator tensor_list() {
    return {
        Tensor(), // aev_t got removed
        atomI_t,
        startIdxJ_t,
        torch::tensor(nI),
        coordinates_t,
        species_t,
        torch::tensor(radialNbr.nJ),
        torch::tensor(radialNbr.maxNumJPerI),
        radialNbr.atomJ_t,
        radialNbr.numJPerI_t,
        radialNbr.distJ_t,
        radialNbr.deltaJ_t,
        torch::tensor(angularNbr.nJ),
        torch::tensor(angularNbr.maxNumJPerI),
        angularNbr.atomJ_t,
        angularNbr.numJPerI_t,
        angularNbr.distJ_t,
        angularNbr.deltaJ_t};
  }
};

// cuda kernels
template <bool use_cos_cutoff>
void cuaev_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params,
    Result& result);

template <bool use_cos_cutoff>
void cuaev_forward_with_half_nbrlist(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomIJ_t,
    const Tensor& deltaJ_t,
    const Tensor& distJ_t,
    const AEVScalarParams& aev_params,
    Result& result);

template <bool use_cos_cutoff>
void cuaev_forward_with_full_nbrlist(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const Tensor& atomI_t,
    const Tensor& atomJ_t,
    const Tensor& numJPerI_t,
    const AEVScalarParams& aev_params,
    Result& result);

template <bool use_cos_cutoff>
Tensor cuaev_backward(const Tensor& grad_output, const AEVScalarParams& aev_params, const Result& result);

template <bool use_cos_cutoff>
Tensor cuaev_double_backward(const Tensor& grad_force, const AEVScalarParams& aev_params, const Result& result);

void initAEVConsts(AEVScalarParams& aev_params, cudaStream_t stream);

// CuaevComputer
// Only keep one copy of aev parameters
struct CuaevComputer : torch::CustomClassHolder {
  AEVScalarParams aev_params;

  CuaevComputer(
      double Rcr,
      double Rca,
      const Tensor& EtaR_t,
      const Tensor& ShfR_t,
      const Tensor& EtaA_t,
      const Tensor& Zeta_t,
      const Tensor& ShfA_t,
      const Tensor& ShfZ_t,
      int64_t num_species,
      bool use_cos_cutoff);

  // TODO add option for simulation only forward, which will initilize result space, and no need to allocate any more.
  Result forward(const Tensor& coordinates_t, const Tensor& species_t);

  Result forward_with_half_nbrlist(
      const Tensor& coordinates_t,
      const Tensor& species_t,
      const Tensor& atomIJ_t,
      const Tensor& deltaJ_t,
      const Tensor& distJ_t);

  Result forward_with_full_nbrlist(
      const Tensor& coordinates_t,
      const Tensor& species_t,
      const Tensor& atomIJ_t,
      const Tensor& deltaJ_t,
      const Tensor& distJ_t);

  Tensor backward(const Tensor& grad_e_aev, const Result& result);

  Tensor double_backward(const Tensor& grad_force, const Result& result);
};

// Autograd functions
class CuaevDoubleAutograd : public torch::autograd::Function<CuaevDoubleAutograd> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor grad_e_aev,
      const torch::intrusive_ptr<CuaevComputer>& cuaev_computer,
      tensor_list result_tensors);
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

class CuaevAutograd : public torch::autograd::Function<CuaevAutograd> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      const Tensor& coordinates_t,
      const Tensor& species_t,
      const torch::intrusive_ptr<CuaevComputer>& cuaev_computer);
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

class CuaevWithHalfNbrlistAutograd : public torch::autograd::Function<CuaevWithHalfNbrlistAutograd> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      const Tensor& coordinates_t,
      const Tensor& species_t,
      const Tensor& atomIJ_t,
      const Tensor& deltaJ_t,
      const Tensor& distJ_t,
      const torch::intrusive_ptr<CuaevComputer>& cuaev_computer);
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

class CuaevWithFullNbrlistAutograd : public torch::autograd::Function<CuaevWithFullNbrlistAutograd> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      const Tensor& coordinates_t,
      const Tensor& species_t,
      const Tensor& atomI_t,
      const Tensor& atomJ_t,
      const Tensor& numJPerI_t,
      const torch::intrusive_ptr<CuaevComputer>& cuaev_computer);
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};
