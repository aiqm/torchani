#ifndef CUAEV_COMPUTER
#define CUAEV_COMPUTER

#include <c10/cuda/CUDACachingAllocator.h>
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

struct AEVScalarParams {
  float Rcr;
  float Rca;
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
};

struct Result : torch::CustomClassHolder {
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
  Tensor coordinates_t;
  Tensor species_t;

  Result();
  Result(
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
      Tensor species_t_);
  void release();
  ~Result() {
    this->release();
  }
};

// cuda kernels
Result cuaev_forward(const Tensor& coordinates_t, const Tensor& species_t, const AEVScalarParams& aev_params);
Tensor cuaev_backward(
    const Tensor& grad_output,
    const AEVScalarParams& aev_params,
    const torch::intrusive_ptr<Result>& res_pt);
Tensor cuaev_double_backward(
    const Tensor& grad_force,
    const AEVScalarParams& aev_params,
    const torch::intrusive_ptr<Result>& res_pt);

// CuaevComputer
// Only keep one copy of aev parameters and one copy of result for backward
struct CuaevComputer : torch::CustomClassHolder {
  AEVScalarParams aev_params;
  Result result;

  CuaevComputer(
      double Rcr,
      double Rca,
      const Tensor& EtaR_t,
      const Tensor& ShfR_t,
      const Tensor& EtaA_t,
      const Tensor& Zeta_t,
      const Tensor& ShfA_t,
      const Tensor& ShfZ_t,
      int64_t num_species);

  Result forward(const Tensor& coordinates_t, const Tensor& species_t);
  Tensor backward(const Tensor& grad_e_aev, const torch::intrusive_ptr<Result>& res_pt);
  Tensor double_backward(const Tensor& grad_force, const torch::intrusive_ptr<Result>& res_pt);
};

// Autograd functions
class CuaevDoubleAutograd : public torch::autograd::Function<CuaevDoubleAutograd> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor grad_e_aev,
      AutogradContext* prectx,
      const torch::intrusive_ptr<Result>& res_pt);
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

#endif
