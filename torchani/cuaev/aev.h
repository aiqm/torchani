#ifndef CUAEV_COMPUTER
#define CUAEV_COMPUTER

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <iostream>
using torch::Tensor;

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

struct Result {
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

  Result() {
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

  void release() {
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

  ~Result() {
    this->release();
  }
};

void cuaev_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params,
    Result& result);

Tensor cuaev_backward(const Tensor& grad_output, const AEVScalarParams& aev_params, Result* res_pt);

Tensor cuaev_double_backward(const Tensor& grad_force, const AEVScalarParams& aev_params, Result* res_pt);

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
    std::cout << "hello"
              << "\n";
  };

  Tensor forward(const Tensor& coordinates_t, const Tensor& species_t) {
    cuaev_forward(coordinates_t, species_t, aev_params, result);
    return result.aev_t;
  };

  Tensor backward(const Tensor& grad_e_aev) {
    Tensor force = cuaev_backward(grad_e_aev, aev_params, &result);
    return force;
  };

  Tensor double_backward(const Tensor& grad_force) {
    Tensor grad_grad_aev = cuaev_double_backward(grad_force, aev_params, &result);
    return grad_grad_aev;
  };
};

#endif
