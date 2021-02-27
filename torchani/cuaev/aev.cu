#include <thrust/equal.h>
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <vector>

#include <ATen/Context.h>
#include <THC/THC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <THC/THCThrustAllocator.cuh>

#define PI 3.141592653589793
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

template <typename DataT, typename IndexT = int>
struct AEVScalarParams {
  DataT Rcr;
  DataT Rca;
  IndexT radial_sublength;
  IndexT radial_length;
  IndexT angular_sublength;
  IndexT angular_length;
  IndexT num_species;

  AEVScalarParams() = default;

  AEVScalarParams(const torch::IValue& aev_params_ivalue) {
    c10::intrusive_ptr<c10::ivalue::Tuple> aev_params_tuple_ptr = aev_params_ivalue.toTuple();
    auto aev_params_tuple = aev_params_tuple_ptr->elements();

    Rcr = static_cast<DataT>(aev_params_tuple[0].toDouble());
    Rca = static_cast<DataT>(aev_params_tuple[1].toDouble());
    radial_sublength = static_cast<IndexT>(aev_params_tuple[2].toInt());
    radial_length = static_cast<IndexT>(aev_params_tuple[3].toInt());
    angular_sublength = static_cast<IndexT>(aev_params_tuple[4].toInt());
    angular_length = static_cast<IndexT>(aev_params_tuple[5].toInt());
    num_species = static_cast<IndexT>(aev_params_tuple[6].toInt());
  }

  operator torch::IValue() {
    return torch::IValue(std::make_tuple(
        (double)Rcr, (double)Rca, radial_sublength, radial_length, angular_sublength, angular_length, num_species));
  }
};

// fetch from the following matrix
// [[ 0,  1,  2,  3,  4],
//  [ 1,  5,  6,  7,  8],
//  [ 2,  6,  9, 10, 11],
//  [ 3,  7, 10, 12, 13],
//  [ 4,  8, 11, 13, 14]]
constexpr int csubaev_offsets(int i, int j, int n) {
  int larger = std::max(i, j);
  int smaller = std::min(i, j);
  int starting = smaller * (2 * n - smaller + 1) / 2; // n + (n - 1) + ... + (n - smaller + 1)
  int offset = larger - smaller;
  return starting + offset;
}

template <typename DataT>
struct PairDist {
  DataT Rij;
  int midx;
  short i;
  short j;
};

// used to group Rijs by atom id
template <typename DataT>
__host__ __device__ bool operator==(const PairDist<DataT>& lhs, const PairDist<DataT>& rhs) {
  return lhs.midx == rhs.midx && lhs.i == rhs.i;
}

/// Alignment of memory. Must be a power of two
/// \tparam boundary Boundary to align to (NOTE: must be power of 2)
/// \param value Input value that is to be aligned
/// \return Value aligned to boundary
template <int32_t boundary>
__host__ __device__ __forceinline__ int align(const int& value) {
  static_assert((boundary & (boundary - 1)) == 0, "Boundary for align must be power of 2");
  return (value + boundary) & ~(boundary - 1);
}

template <typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    PairDist<DataT>* d_Rij,
    IndexT max_natoms_per_mol) {
  extern __shared__ DataT spos[];
  DataT* sx = &spos[0];
  DataT* sy = &spos[max_natoms_per_mol];
  DataT* sz = &spos[2 * max_natoms_per_mol];

  int mol_idx = blockIdx.x;
  int tidx = threadIdx.y * blockDim.x + threadIdx.x;

  for (int i = tidx; i < max_natoms_per_mol; i += blockDim.x * blockDim.y) {
    sx[i] = pos_t[mol_idx][i][0];
    sy[i] = pos_t[mol_idx][i][1];
    sz[i] = pos_t[mol_idx][i][2];
  }

  __syncthreads();

  int natom_pairs = max_natoms_per_mol * max_natoms_per_mol;

  for (int i = threadIdx.y; i < max_natoms_per_mol; i += blockDim.y) {
    SpeciesT type_i = species_t[mol_idx][i];

    DataT xi = sx[i];
    DataT yi = sy[i];
    DataT zi = sz[i];

    for (int j = threadIdx.x; j < max_natoms_per_mol; j += blockDim.x) {
      SpeciesT type_j = species_t[mol_idx][j];

      const DataT xj = sx[j];
      const DataT yj = sy[j];
      const DataT zj = sz[j];
      const DataT delx = xj - xi;
      const DataT dely = yj - yi;
      const DataT delz = zj - zi;

      const DataT Rsq = delx * delx + dely * dely + delz * delz;
      if (type_i != -1 && type_j != -1 && i != j) {
        DataT Rij = sqrt(Rsq);

        PairDist<DataT> d;
        d.Rij = Rij;
        d.midx = mol_idx;
        d.i = i;
        d.j = j;

        d_Rij[mol_idx * natom_pairs + i * max_natoms_per_mol + j] = d;
      }
    }
  }
}

template <typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistanceSingleMolecule(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    PairDist<DataT>* d_Rij,
    IndexT max_natoms_per_mol) {
  constexpr int mol_idx = 0;
  int natom_pairs = max_natoms_per_mol * max_natoms_per_mol;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= max_natoms_per_mol || j >= max_natoms_per_mol)
    return;

  SpeciesT type_i = species_t[mol_idx][i];
  DataT xi = pos_t[mol_idx][i][0];
  DataT yi = pos_t[mol_idx][i][1];
  DataT zi = pos_t[mol_idx][i][2];

  SpeciesT type_j = species_t[mol_idx][j];
  DataT xj = pos_t[mol_idx][j][0];
  DataT yj = pos_t[mol_idx][j][1];
  DataT zj = pos_t[mol_idx][j][2];

  DataT delx = xj - xi;
  DataT dely = yj - yi;
  DataT delz = zj - zi;

  DataT Rsq = delx * delx + dely * dely + delz * delz;

  if (type_i != -1 && type_j != -1 && i != j) {
    DataT Rij = sqrt(Rsq);

    PairDist<DataT> d;
    d.Rij = Rij;
    d.midx = mol_idx;
    d.i = i;
    d.j = j;

    d_Rij[mol_idx * natom_pairs + i * max_natoms_per_mol + j] = d;
  }
}

// every block compute blocksize RIJ's gradient by column major, to avoid atomicAdd waiting
template <bool is_double_backward, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance_backward_or_doublebackward(
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits>
        grad_dist, // ddist for backward, dddist for double backward
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_coord_or_force, // dcoord for backward, dforce(i.e. ddcoord) for double backward
    const PairDist<DataT>* d_radialRij,
    IndexT nRadialRij) {
  int gidx = threadIdx.x * gridDim.x + blockIdx.x;

  if (gidx >= nRadialRij)
    return;

  PairDist<DataT> d = d_radialRij[gidx];
  DataT Rij = d.Rij;
  int mol_idx = d.midx;
  int i = d.i;
  int j = d.j;

  const DataT delx = pos_t[mol_idx][j][0] - pos_t[mol_idx][i][0];
  const DataT dely = pos_t[mol_idx][j][1] - pos_t[mol_idx][i][1];
  const DataT delz = pos_t[mol_idx][j][2] - pos_t[mol_idx][i][2];

  if constexpr (is_double_backward) {
    auto& grad_force = grad_coord_or_force;
    DataT grad_force_coord_Rij_item = (grad_force[mol_idx][j][0] - grad_force[mol_idx][i][0]) * delx / Rij +
        (grad_force[mol_idx][j][1] - grad_force[mol_idx][i][1]) * dely / Rij +
        (grad_force[mol_idx][j][2] - grad_force[mol_idx][i][2]) * delz / Rij;

    grad_dist[gidx] = grad_force_coord_Rij_item;
  } else {
    auto& grad_coord = grad_coord_or_force;

    DataT grad_dist_coord_x = delx / Rij;
    DataT grad_dist_coord_y = dely / Rij;
    DataT grad_dist_coord_z = delz / Rij;
    DataT grad_radial_dist_item = grad_dist[gidx];

    atomicAdd(&grad_coord[mol_idx][j][0], grad_radial_dist_item * grad_dist_coord_x);
    atomicAdd(&grad_coord[mol_idx][j][1], grad_radial_dist_item * grad_dist_coord_y);
    atomicAdd(&grad_coord[mol_idx][j][2], grad_radial_dist_item * grad_dist_coord_z);
    atomicAdd(&grad_coord[mol_idx][i][0], -grad_radial_dist_item * grad_dist_coord_x);
    atomicAdd(&grad_coord[mol_idx][i][1], -grad_radial_dist_item * grad_dist_coord_y);
    atomicAdd(&grad_coord[mol_idx][i][2], -grad_radial_dist_item * grad_dist_coord_z);
  }
}

template <typename SpeciesT, typename DataT, typename IndexT = int, int TILEX = 8, int TILEY = 4>
__global__ void cuAngularAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    PairDist<DataT>* d_Rij,
    PairDist<DataT>* d_centralAtom,
    int* d_nPairsPerCenterAtom,
    int* d_centerAtomStartIdx,
    AEVScalarParams<DataT, IndexT> aev_params,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned,
    int ncentral_atoms) {
  extern __shared__ DataT smem[];

  constexpr int threads_per_catom = TILEX * TILEY;
  static_assert(threads_per_catom == C10_WARP_SIZE);
  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = gIdx / threads_per_catom; // central atom id

  if (cIdx >= ncentral_atoms)
    return;

  int groupIdx = threadIdx.x / threads_per_catom;
  int laneIdx = threadIdx.x % threads_per_catom;
  int ncatom_per_tpb = blockDim.x / threads_per_catom;

  DataT* saev = &smem[groupIdx * angular_length_aligned];

  int offset = ncatom_per_tpb * angular_length_aligned;
  DataT* sdx = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT* sdy = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT* sdz = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT* sdist = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT* sfc = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  int* stype = (int*)&smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);
  DataT Rca = aev_params.Rca;
  IndexT num_species = aev_params.num_species;

  PairDist<DataT> d = d_centralAtom[cIdx];
  int start_idx = d_centerAtomStartIdx[cIdx];
  int jnum = d_nPairsPerCenterAtom[cIdx];

  // center atom
  int i = d.i;
  int mol_idx = d.midx;

  for (int iaev = laneIdx; iaev < aev_params.angular_length; iaev += threads_per_catom) {
    saev[iaev] = 0;
  }

  DataT xi = pos_t[mol_idx][i][0];
  DataT yi = pos_t[mol_idx][i][1];
  DataT zi = pos_t[mol_idx][i][2];

  for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
    PairDist<DataT> dij = d_Rij[start_idx + jj];
    int j = dij.j;
    DataT Rij = dij.Rij;
    SpeciesT type_j = species_t[mol_idx][j];
    sdx[jj] = pos_t[mol_idx][j][0] - xi;
    sdy[jj] = pos_t[mol_idx][j][1] - yi;
    sdz[jj] = pos_t[mol_idx][j][2] - zi;
    stype[jj] = type_j;
    sdist[jj] = Rij;
    DataT fc_ij = 0.5 * cos(PI * Rij / Rca) + 0.5;
    sfc[jj] = fc_ij;
  }

  short2 tile = make_short2(laneIdx % TILEX, laneIdx / TILEX);
  // must sync if threads_per_catom != 32 (wrap size) to make sure shared data is ready
  // __syncthreads

  for (int jj = 0; jj < jnum; jj++) {
    const DataT Rij = sdist[jj];
    SpeciesT type_j = stype[jj];

    DataT fc_ij = sfc[jj];

    for (int kk_start = jj + 1; kk_start < jnum; kk_start += threads_per_catom) {
      int kk = kk_start + laneIdx;
      DataT theta = 0;
      if (kk < jnum) {
        const DataT Rik = sdist[kk];
        theta = acos(0.95 * (sdx[jj] * sdx[kk] + sdy[jj] * sdy[kk] + sdz[jj] * sdz[kk]) / (Rij * Rik));
      }

      for (int srcLane = 0; srcLane < C10_WARP_SIZE && (kk_start + srcLane) < jnum; ++srcLane) {
        int kk = kk_start + srcLane;
        DataT theta_ijk = __shfl_sync(0xFFFFFFFF, theta, srcLane);

        const DataT Rik = sdist[kk];
        SpeciesT type_k = stype[kk];

        DataT fc_ik = sfc[kk];

        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = aev_params.angular_sublength * csubaev_offsets(type_j, type_k, num_species);

        for (int itheta = tile.x; itheta < nShfZ; itheta += TILEX) {
          DataT ShfZ = ShfZ_t[itheta];

          DataT factor1 = pow((1 + cos(theta_ijk - ShfZ)) / 2, Zeta);

          for (int ishfr = tile.y; ishfr < nShfA; ishfr += TILEY) {
            DataT ShfA = ShfA_t[ishfr];
            DataT factor2 = exp(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));

            DataT res = 2 * factor1 * factor2 * fc_ijk;

            saev[subaev_offset + ishfr * nShfZ + itheta] += res;
          }
        }
      }
    }
  }

  for (int iaev = laneIdx; iaev < aev_params.angular_length; iaev += threads_per_catom) {
    aev_t[mol_idx][i][aev_params.radial_length + iaev] = saev[iaev];
  }
}

template <
    bool is_double_backward,
    typename SpeciesT,
    typename DataT,
    typename IndexT = int,
    int TILEX = 8,
    int TILEY = 4>
__global__ void cuAngularAEVs_backward_or_doublebackward(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_output, // for backward, this is daev, for double backward, this is dforce (i.e. ddcoord)
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_input, // for backward, this is dcoord, for double backward, this is ddaev
    const PairDist<DataT>* d_Rij,
    const PairDist<DataT>* d_centralAtom,
    int* d_nPairsPerCenterAtom,
    int* d_centerAtomStartIdx,
    AEVScalarParams<DataT, IndexT> aev_params,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned,
    int ncentral_atoms) {
  extern __shared__ DataT smem[];

  constexpr int threads_per_catom = TILEX * TILEY;
  static_assert(threads_per_catom == C10_WARP_SIZE);
  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = gIdx / threads_per_catom; // central atom id

  if (cIdx >= ncentral_atoms)
    return;

  int groupIdx = threadIdx.x / threads_per_catom;
  int laneIdx = threadIdx.x % threads_per_catom;
  int ncatom_per_tpb = blockDim.x / threads_per_catom; // e.g. 2 catom per block

  DataT* sdx = &smem[groupIdx * maxnbrs_per_atom_aligned];
  int offset = ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdy = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdz = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjx_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjy_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdjz_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sdist = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sfc = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  DataT* sfc_grad = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];
  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;

  int* stype = (int*)&smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);
  DataT Rca = aev_params.Rca;
  IndexT num_species = aev_params.num_species;

  PairDist<DataT> d = d_centralAtom[cIdx];
  int start_idx = d_centerAtomStartIdx[cIdx];
  int jnum = d_nPairsPerCenterAtom[cIdx];

  // center atom
  int i = d.i;
  int mol_idx = d.midx;

  DataT xi = pos_t[mol_idx][i][0];
  DataT yi = pos_t[mol_idx][i][1];
  DataT zi = pos_t[mol_idx][i][2];

  for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
    PairDist<DataT> dij = d_Rij[start_idx + jj];
    int j = dij.j;
    DataT Rij = dij.Rij;
    SpeciesT type_j = species_t[mol_idx][j];
    sdx[jj] = pos_t[mol_idx][j][0] - xi;
    sdy[jj] = pos_t[mol_idx][j][1] - yi;
    sdz[jj] = pos_t[mol_idx][j][2] - zi;
    stype[jj] = type_j;
    sdist[jj] = Rij;
    // cutoff
    DataT fc_ij = 0.5 * cos(PI * Rij / Rca) + 0.5;
    DataT fc_ij_grad = -0.5 * (PI / Rca) * sin(PI * Rij / Rca);
    sfc[jj] = fc_ij;
    sfc_grad[jj] = fc_ij_grad;
  }

  // grad init
  DataT sdix_grad = 0;
  DataT sdiy_grad = 0;
  DataT sdiz_grad = 0;

  for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
    sdjx_grad[jj] = 0;
    sdjy_grad[jj] = 0;
    sdjz_grad[jj] = 0;
  }

  short2 tile = make_short2(laneIdx % TILEX, laneIdx / TILEX);
  const DataT tc = 0.95; // theta constant factor
  // must sync if threads_per_catom != 32 (wrap size) to make sure shared data is ready
  // __syncthreads

  for (int jj = 0; jj < jnum; jj++) {
    const DataT Rij = sdist[jj];
    SpeciesT type_j = stype[jj];

    DataT fc_ij = sfc[jj];
    DataT grad_fc_ij = sfc_grad[jj];

    for (int kk_start = jj + 1; kk_start < jnum; kk_start += threads_per_catom) {
      int kk = kk_start + laneIdx;
      DataT theta = 0;
      DataT grad_theta_vij_x = 0;
      DataT grad_theta_vij_y = 0;
      DataT grad_theta_vij_z = 0;
      DataT grad_theta_vik_x = 0;
      DataT grad_theta_vik_y = 0;
      DataT grad_theta_vik_z = 0;
      if (kk < jnum) {
        const DataT Rik = sdist[kk];
        DataT vij_vik_dot = sdx[jj] * sdx[kk] + sdy[jj] * sdy[kk] + sdz[jj] * sdz[kk];
        theta = acos(tc * vij_vik_dot / (Rij * Rik));
        // grad
        DataT vij_factor =
            tc / (Rij * Rij * Rij * sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rij * Rij) + Rik * Rik));
        DataT vik_factor = tc /
            (Rik * Rik * Rik *
             sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rik * Rik) + Rij * Rij)); // tricky 80ms improved
        grad_theta_vij_x = vij_factor * (sdx[jj] * vij_vik_dot - sdx[kk] * Rij * Rij);
        grad_theta_vij_y = vij_factor * (sdy[jj] * vij_vik_dot - sdy[kk] * Rij * Rij);
        grad_theta_vij_z = vij_factor * (sdz[jj] * vij_vik_dot - sdz[kk] * Rij * Rij);
        grad_theta_vik_x = vik_factor * (sdx[kk] * vij_vik_dot - sdx[jj] * Rik * Rik);
        grad_theta_vik_y = vik_factor * (sdy[kk] * vij_vik_dot - sdy[jj] * Rik * Rik);
        grad_theta_vik_z = vik_factor * (sdz[kk] * vij_vik_dot - sdz[jj] * Rik * Rik);
      }

      for (int srcLane = 0; srcLane < C10_WARP_SIZE && (kk_start + srcLane) < jnum; ++srcLane) {
        int kk = kk_start + srcLane;
        DataT theta_ijk = __shfl_sync(0xFFFFFFFF, theta, srcLane);
        // TODO necessary?
        DataT grad_theta_vij_x_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_x, srcLane);
        DataT grad_theta_vij_y_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_y, srcLane);
        DataT grad_theta_vij_z_ = __shfl_sync(0xFFFFFFFF, grad_theta_vij_z, srcLane);
        DataT grad_theta_vik_x_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_x, srcLane);
        DataT grad_theta_vik_y_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_y, srcLane);
        DataT grad_theta_vik_z_ = __shfl_sync(0xFFFFFFFF, grad_theta_vik_z, srcLane);

        const DataT Rik = sdist[kk];
        SpeciesT type_k = stype[kk];

        DataT fc_ik = sfc[kk];
        DataT grad_fc_ik = sfc_grad[kk];

        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = aev_params.angular_sublength * csubaev_offsets(type_j, type_k, num_species);

        for (int itheta = tile.x; itheta < nShfZ; itheta += TILEX) {
          DataT ShfZ = ShfZ_t[itheta];

          DataT factor1 = pow((1 + cos(theta_ijk - ShfZ)) / 2, Zeta);
          DataT grad_factor1_theta = 1.0 / 2.0 * Zeta * pow((1 + cos(ShfZ - theta_ijk)) / 2, Zeta - 1) *
              sin(ShfZ - theta_ijk); // tricky 100ms improved

          for (int ishfr = tile.y; ishfr < nShfA; ishfr += TILEY) {
            DataT ShfA = ShfA_t[ishfr];
            DataT factor2 = exp(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));
            DataT grad_factor2_dist = -EtaA * (Rijk - ShfA) * factor2;

            DataT grad_vij_x = 2 *
                (grad_factor1_theta * grad_theta_vij_x_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdx[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdx[jj] / Rij);
            DataT grad_vij_y = 2 *
                (grad_factor1_theta * grad_theta_vij_y_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdy[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdy[jj] / Rij);
            DataT grad_vij_z = 2 *
                (grad_factor1_theta * grad_theta_vij_z_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdz[jj] / Rij * fc_ijk +
                 factor1 * factor2 * fc_ik * grad_fc_ij * sdz[jj] / Rij);
            DataT grad_vik_x = 2 *
                (grad_factor1_theta * grad_theta_vik_x_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdx[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdx[kk] / Rik);
            DataT grad_vik_y = 2 *
                (grad_factor1_theta * grad_theta_vik_y_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdy[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdy[kk] / Rik);
            DataT grad_vik_z = 2 *
                (grad_factor1_theta * grad_theta_vik_z_ * factor2 * fc_ijk +
                 factor1 * grad_factor2_dist * sdz[kk] / Rik * fc_ijk +
                 factor1 * factor2 * fc_ij * grad_fc_ik * sdz[kk] / Rik);

            if constexpr (is_double_backward) {
              int atomj_idx = d_Rij[start_idx + jj].j;
              int atomk_idx = d_Rij[start_idx + kk].j;
              auto& grad_force = grad_output;
              auto& grad_grad_aev = grad_input;
              grad_vij_x *= (grad_force[mol_idx][atomj_idx][0] - grad_force[mol_idx][i][0]);
              grad_vij_y *= (grad_force[mol_idx][atomj_idx][1] - grad_force[mol_idx][i][1]);
              grad_vij_z *= (grad_force[mol_idx][atomj_idx][2] - grad_force[mol_idx][i][2]);
              grad_vik_x *= (grad_force[mol_idx][atomk_idx][0] - grad_force[mol_idx][i][0]);
              grad_vik_y *= (grad_force[mol_idx][atomk_idx][1] - grad_force[mol_idx][i][1]);
              grad_vik_z *= (grad_force[mol_idx][atomk_idx][2] - grad_force[mol_idx][i][2]);
              atomicAdd(
                  &grad_grad_aev[mol_idx][i][aev_params.radial_length + subaev_offset + ishfr * nShfZ + itheta],
                  grad_vij_x + grad_vij_y + grad_vij_z + grad_vik_x + grad_vik_y + grad_vik_z);
            } else {
              DataT grad_output_item =
                  grad_output[mol_idx][i][aev_params.radial_length + subaev_offset + ishfr * nShfZ + itheta];
              grad_vij_x *= grad_output_item;
              grad_vij_y *= grad_output_item;
              grad_vij_z *= grad_output_item;
              grad_vik_x *= grad_output_item;
              grad_vik_y *= grad_output_item;
              grad_vik_z *= grad_output_item;

              sdix_grad += (-grad_vij_x - grad_vik_x);
              sdiy_grad += (-grad_vij_y - grad_vik_y);
              sdiz_grad += (-grad_vij_z - grad_vik_z);

              for (int offset = 16; offset > 0; offset /= 2) {
                grad_vij_x += __shfl_down_sync(0xFFFFFFFF, grad_vij_x, offset);
                grad_vij_y += __shfl_down_sync(0xFFFFFFFF, grad_vij_y, offset);
                grad_vij_z += __shfl_down_sync(0xFFFFFFFF, grad_vij_z, offset);
                grad_vik_x += __shfl_down_sync(0xFFFFFFFF, grad_vik_x, offset);
                grad_vik_y += __shfl_down_sync(0xFFFFFFFF, grad_vik_y, offset);
                grad_vik_z += __shfl_down_sync(0xFFFFFFFF, grad_vik_z, offset);
              }
              if (laneIdx == 0) {
                sdjx_grad[jj] += grad_vij_x;
                sdjy_grad[jj] += grad_vij_y;
                sdjz_grad[jj] += grad_vij_z;

                sdjx_grad[kk] += grad_vik_x;
                sdjy_grad[kk] += grad_vik_y;
                sdjz_grad[kk] += grad_vik_z;
              }
            }
          }
        }
      }
    }
  }

  if constexpr (!is_double_backward) {
    auto& grad_coord = grad_input;
    int atomi_idx = i;
    atomicAdd(&grad_coord[mol_idx][atomi_idx][0], sdix_grad);
    atomicAdd(&grad_coord[mol_idx][atomi_idx][1], sdiy_grad);
    atomicAdd(&grad_coord[mol_idx][atomi_idx][2], sdiz_grad);

    for (int jj = laneIdx; jj < jnum; jj += threads_per_catom) {
      int atomj_idx = d_Rij[start_idx + jj].j;

      atomicAdd(&grad_coord[mol_idx][atomj_idx][0], sdjx_grad[jj]);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][1], sdjy_grad[jj]);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][2], sdjz_grad[jj]);
    }
  }
}

template <typename SpeciesT, typename DataT, int THREADS_PER_RIJ>
__global__ void cuRadialAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    PairDist<DataT>* d_Rij,
    AEVScalarParams<DataT, int> aev_params,
    int nRadialRij) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = gidx / THREADS_PER_RIJ;

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  if (idx >= nRadialRij)
    return;

  int laneIdx = threadIdx.x % THREADS_PER_RIJ;

  PairDist<DataT> d = d_Rij[idx];
  DataT Rij = d.Rij;
  int mol_idx = d.midx;
  int i = d.i;
  int j = d.j;

  SpeciesT type_j = species_t[mol_idx][j];

  DataT fc = 0.5 * cos(PI * Rij / aev_params.Rcr) + 0.5;

  for (int ishfr = laneIdx; ishfr < nShfR; ishfr += THREADS_PER_RIJ) {
    DataT ShfR = ShfR_t[ishfr];

    DataT GmR = 0.25 * exp(-EtaR * (Rij - ShfR) * (Rij - ShfR)) * fc;

    atomicAdd(&aev_t[mol_idx][i][type_j * aev_params.radial_sublength + ishfr], GmR);
  }
}

// every <THREADS_PER_RIJ> threads take care of 1 RIJ, and iterate <nShfR / THREADS_PER_RIJ> times
template <bool is_double_backward, typename SpeciesT, typename DataT, int THREADS_PER_RIJ>
__global__ void cuRadialAEVs_backward_or_doublebackward(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_aev, // daev for backward, ddaev for double backward
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits>
        grad_dist, // ddist for backward, dddist for double backward
    const PairDist<DataT>* d_Rij,
    AEVScalarParams<DataT, int> aev_params,
    int nRadialRij) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = gidx / THREADS_PER_RIJ;

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  if (idx >= nRadialRij)
    return;

  int laneIdx = threadIdx.x % THREADS_PER_RIJ;

  PairDist<DataT> d = d_Rij[idx];
  DataT Rij = d.Rij;
  int mol_idx = d.midx;
  int i = d.i;
  int j = d.j;

  SpeciesT type_j = species_t[mol_idx][j];

  DataT fc = 0.5 * cos(PI * Rij / aev_params.Rcr) + 0.5;
  DataT fc_grad = -0.5 * (PI / aev_params.Rcr) * sin(PI * Rij / aev_params.Rcr);

  DataT upstream_grad;
  if constexpr (is_double_backward) {
    upstream_grad = grad_dist[idx];
  }

  for (int ishfr = laneIdx; ishfr < nShfR; ishfr += THREADS_PER_RIJ) {
    DataT ShfR = ShfR_t[ishfr];

    DataT GmR = 0.25 * exp(-EtaR * (Rij - ShfR) * (Rij - ShfR));
    DataT GmR_grad = -EtaR * (-2 * ShfR + 2 * Rij) * GmR;
    DataT jacobian = GmR_grad * fc + GmR * fc_grad;

    if constexpr (is_double_backward) {
      atomicAdd(&grad_aev[mol_idx][i][type_j * aev_params.radial_sublength + ishfr], upstream_grad * jacobian);
    } else {
      upstream_grad = grad_aev[mol_idx][i][type_j * aev_params.radial_sublength + ishfr];
      atomicAdd(&grad_dist[idx], upstream_grad * jacobian);
    }
  }
}

template <typename DataT>
void cubScan(const DataT* d_in, DataT* d_out, int num_items, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

template <typename DataT, typename IndexT>
int cubEncode(
    const DataT* d_in,
    DataT* d_unique_out,
    IndexT* d_counts_out,
    int num_items,
    int* d_num_runs_out,
    cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run encoding
  cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);

  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_runs_out, sizeof(int), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  return num_selected;
}

template <typename DataT, typename LambdaOpT>
int cubDeviceSelect(
    const DataT* d_in,
    DataT* d_out,
    int num_items,
    int* d_num_selected_out,
    LambdaOpT select_op,
    cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run selection
  cub::DeviceSelect::If(
      d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op, stream);

  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return num_selected;
}

template <typename DataT>
DataT cubMax(const DataT* d_in, int num_items, DataT* d_out, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run min-reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  int maxVal = 0;
  cudaMemcpyAsync(&maxVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return maxVal;
}

struct Result {
  Tensor aev_t;
  AEVScalarParams<float> aev_params;
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
};

// NOTE: assumes size of EtaA_t = Zeta_t = EtaR_t = 1
template <typename ScalarRealT = float>
Result cuaev_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    double Rcr_,
    double Rca_,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species_) {
  TORCH_CHECK(
      (species_t.dtype() == torch::kInt32) && (coordinates_t.dtype() == torch::kFloat32), "Unsupported input type");
  TORCH_CHECK(
      EtaR_t.size(0) == 1 || EtaA_t.size(0) == 1 || Zeta_t.size(0) == 1,
      "cuda extension is currently not supported for the specified "
      "configuration");

  ScalarRealT Rcr = Rcr_;
  ScalarRealT Rca = Rca_;
  int num_species = num_species_;

  const int n_molecules = species_t.size(0);
  const int max_natoms_per_mol = species_t.size(1);

  AEVScalarParams<float> aev_params;
  aev_params.Rca = Rca;
  aev_params.Rcr = Rcr;
  aev_params.num_species = num_species;

  aev_params.radial_sublength = EtaR_t.size(0) * ShfR_t.size(0);
  aev_params.radial_length = aev_params.radial_sublength * num_species;

  aev_params.angular_sublength = EtaA_t.size(0) * Zeta_t.size(0) * ShfA_t.size(0) * ShfZ_t.size(0);
  aev_params.angular_length = aev_params.angular_sublength * (num_species * (num_species + 1) / 2);

  int aev_length = aev_params.radial_length + aev_params.angular_length;

  auto aev_t = torch::zeros({n_molecules, max_natoms_per_mol, aev_length}, coordinates_t.options());

  if (species_t.numel() == 0) {
    return {aev_t, aev_params, Tensor(), Tensor(), Tensor(), 0, 0, 0};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto thrust_allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(thrust_allocator).on(stream);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // buffer to store all the pairwise distance (Rij)
  auto total_natom_pairs = n_molecules * max_natoms_per_mol * max_natoms_per_mol;
  auto d_options = torch::dtype(torch::kUInt8).device(coordinates_t.device());
  Tensor tensor_Rij = torch::empty(sizeof(PairDist<float>) * total_natom_pairs, d_options);
  PairDist<float>* d_Rij = (PairDist<float>*)tensor_Rij.data_ptr();

  // init all Rij to inf
  PairDist<float> init;
  init.Rij = std::numeric_limits<float>::infinity();
  thrust::fill(policy, d_Rij, d_Rij + total_natom_pairs, init);

  // buffer to store all the pairwise distance that is needed for Radial AEV
  // computation
  Tensor tensor_radialRij = torch::empty(sizeof(PairDist<float>) * total_natom_pairs, d_options);
  PairDist<float>* d_radialRij = (PairDist<float>*)tensor_radialRij.data_ptr();

  auto buffer_count = allocator.allocate(sizeof(int));
  int* d_count_out = (int*)buffer_count.get();

  const int block_size = 64;

  if (n_molecules == 1) {
    int tileWidth = 32;
    int tilesPerRow = (max_natoms_per_mol + tileWidth - 1) / tileWidth;
    dim3 block(tileWidth, tileWidth, 1);
    dim3 grid(tilesPerRow, tilesPerRow, 1);
    pairwiseDistanceSingleMolecule<<<grid, block, 0, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_Rij,
        max_natoms_per_mol);
  } else {
    dim3 block(8, 8, 1);
    // Compute pairwise distance (Rij) for all atom pairs in a molecule
    // maximum 4096 atoms, which needs 49152 byte (48 kb) of shared memory
    // TODO: the kernel is not optimized for batched huge molecule (max_natoms_per_mol > 1000)
    pairwiseDistance<<<n_molecules, block, sizeof(float) * max_natoms_per_mol * 3, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_Rij,
        max_natoms_per_mol);
  }

  // Extract Rijs that is needed for RadialAEV comptuation i.e. all the Rij <= Rcr
  int nRadialRij = cubDeviceSelect(
      d_Rij,
      d_radialRij,
      total_natom_pairs,
      d_count_out,
      [=] __device__(const PairDist<float> d) { return d.Rij <= Rcr; },
      stream);

  int nblocks = (nRadialRij * 8 + block_size - 1) / block_size;
  cuRadialAEVs<int, float, 8><<<nblocks, block_size, 0, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_radialRij,
      aev_params,
      nRadialRij);

  // reuse buffer allocated for all Rij
  // d_angularRij will store all the Rij required in Angular AEV computation
  Tensor tensor_angularRij = torch::empty(sizeof(PairDist<float>) * nRadialRij, d_options);
  PairDist<float>* d_angularRij = (PairDist<float>*)tensor_angularRij.data_ptr();

  // Extract Rijs that is needed for AngularAEV comptuation i.e. all the Rij
  // <= Rca
  int nAngularRij = cubDeviceSelect(
      d_radialRij,
      d_angularRij,
      nRadialRij,
      d_count_out,
      [=] __device__(const PairDist<float> d) { return d.Rij <= Rca; },
      stream);

  Tensor tensor_centralAtom = torch::empty(sizeof(PairDist<float>) * nAngularRij, d_options);
  PairDist<float>* d_centralAtom = (PairDist<float>*)tensor_centralAtom.data_ptr();

  Tensor tensor_numPairsPerCenterAtom = torch::empty(sizeof(int) * nAngularRij, d_options);
  int* d_numPairsPerCenterAtom = (int*)tensor_numPairsPerCenterAtom.data_ptr();

  // group by center atom
  int ncenter_atoms = cubEncode(d_angularRij, d_centralAtom, d_numPairsPerCenterAtom, nAngularRij, d_count_out, stream);

  Tensor tensor_centerAtomStartIdx = torch::empty(sizeof(int) * ncenter_atoms, d_options);
  int* d_centerAtomStartIdx = (int*)tensor_centerAtomStartIdx.data_ptr();

  cubScan(d_numPairsPerCenterAtom, d_centerAtomStartIdx, ncenter_atoms, stream);
  {
    const int nthreads_per_catom = 32;
    const int nblocks_angAEV = (ncenter_atoms * nthreads_per_catom + block_size - 1) / block_size;
    auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
      int sm_aev = sizeof(float) * align<4>(aev_params.angular_length); // (angular_length / 4 + 1) * 4
      int sxyz = sizeof(float) * max_nbrs * 3;
      int sRij = sizeof(float) * max_nbrs;
      int sfc = sizeof(float) * max_nbrs;
      int sj = sizeof(int) * max_nbrs;

      return (sm_aev + sxyz + sRij + sfc + sj) * ncatom_per_tpb;
    };

    int maxNbrsPerCenterAtom = cubMax(d_numPairsPerCenterAtom, ncenter_atoms, d_count_out, stream);
    int maxnbrs_per_atom_aligned = align<4>(maxNbrsPerCenterAtom);
    int smem_size_aligned = smem_size(maxnbrs_per_atom_aligned, block_size / nthreads_per_catom);
    int angular_length_aligned = align<4>(aev_params.angular_length);

    cuAngularAEVs<<<nblocks_angAEV, block_size, smem_size_aligned, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_angularRij,
        d_centralAtom,
        d_numPairsPerCenterAtom,
        d_centerAtomStartIdx,
        aev_params,
        maxnbrs_per_atom_aligned,
        angular_length_aligned,
        ncenter_atoms);

    return {aev_t,
            aev_params,
            tensor_Rij,
            tensor_radialRij,
            tensor_angularRij,
            total_natom_pairs,
            nRadialRij,
            nAngularRij,
            tensor_centralAtom,
            tensor_numPairsPerCenterAtom,
            tensor_centerAtomStartIdx,
            maxnbrs_per_atom_aligned,
            angular_length_aligned,
            ncenter_atoms};
  }
}

Tensor cuaev_backward(
    const Tensor& grad_output,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams<float>& aev_params,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    const Tensor& tensor_Rij,
    int total_natom_pairs,
    const Tensor& tensor_radialRij,
    int nRadialRij,
    const Tensor& tensor_angularRij,
    int nAngularRij,
    const Tensor& tensor_centralAtom,
    const Tensor& tensor_numPairsPerCenterAtom,
    const Tensor& tensor_centerAtomStartIdx,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned,
    int ncenter_atoms) {
  using namespace torch::indexing;
  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto grad_coord = torch::zeros(coordinates_t.sizes(), coordinates_t.options().requires_grad(false)); // [2, 5, 3]
  auto grad_output_radial = grad_output.index({Ellipsis, Slice(None, aev_params.radial_length)}); // [2, 5, 64]
  auto grad_output_angular = grad_output.index({Ellipsis, Slice(aev_params.radial_length, None)}); // [2, 5, 320]

  PairDist<float>* d_Rij = (PairDist<float>*)tensor_Rij.data_ptr();
  PairDist<float>* d_radialRij = (PairDist<float>*)tensor_radialRij.data_ptr();
  PairDist<float>* d_angularRij = (PairDist<float>*)tensor_angularRij.data_ptr();
  PairDist<float>* d_centralAtom = (PairDist<float>*)tensor_centralAtom.data_ptr();
  int* d_numPairsPerCenterAtom = (int*)tensor_numPairsPerCenterAtom.data_ptr();
  int* d_centerAtomStartIdx = (int*)tensor_centerAtomStartIdx.data_ptr();

  Tensor grad_radial_dist = torch::zeros(nRadialRij, coordinates_t.options().requires_grad(false));

  int block_size = 64;
  int nblocks = (nRadialRij * 8 + block_size - 1) / block_size;
  cuRadialAEVs_backward_or_doublebackward<false, int, float, 8><<<nblocks, block_size, 0, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_radial_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      d_radialRij,
      aev_params,
      nRadialRij);

  // For best result, block_size should match average molecule size (no padding) to avoid atomicAdd
  nblocks = (nRadialRij + block_size - 1) / block_size;
  pairwiseDistance_backward_or_doublebackward<false><<<nblocks, block_size, 0, stream>>>(
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_radial_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_radialRij,
      nRadialRij);

  auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;

    return (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };

  block_size = 32;
  const int nthreads_per_catom = 32;
  const int nblocks_angAEV = (ncenter_atoms * nthreads_per_catom + block_size - 1) / block_size;
  int smem_size_aligned = smem_size(maxnbrs_per_atom_aligned, block_size / nthreads_per_catom);

  Tensor grad_angular_coord = torch::zeros({nAngularRij, 3}, coordinates_t.options().requires_grad(false));
  cuAngularAEVs_backward_or_doublebackward<false><<<nblocks_angAEV, block_size, smem_size_aligned, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_angularRij,
      d_centralAtom,
      d_numPairsPerCenterAtom,
      d_centerAtomStartIdx,
      aev_params,
      maxnbrs_per_atom_aligned,
      angular_length_aligned,
      ncenter_atoms);

  return grad_coord;
}

Tensor cuaev_double_backward(
    const Tensor& grad_force,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams<float>& aev_params,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    const Tensor& tensor_Rij,
    int total_natom_pairs,
    const Tensor& tensor_radialRij,
    int nRadialRij,
    const Tensor& tensor_angularRij,
    int nAngularRij,
    const Tensor& tensor_centralAtom,
    const Tensor& tensor_numPairsPerCenterAtom,
    const Tensor& tensor_centerAtomStartIdx,
    int maxnbrs_per_atom_aligned,
    int angular_length_aligned,
    int ncenter_atoms) {
  using namespace torch::indexing;
  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int aev_length = aev_params.radial_length + aev_params.angular_length;

  auto grad_grad_aev = torch::zeros(
      {coordinates_t.size(0), coordinates_t.size(1), aev_length},
      coordinates_t.options().requires_grad(false)); // [2, 5, 384]

  PairDist<float>* d_Rij = (PairDist<float>*)tensor_Rij.data_ptr();
  PairDist<float>* d_radialRij = (PairDist<float>*)tensor_radialRij.data_ptr();
  PairDist<float>* d_angularRij = (PairDist<float>*)tensor_angularRij.data_ptr();
  PairDist<float>* d_centralAtom = (PairDist<float>*)tensor_centralAtom.data_ptr();
  int* d_numPairsPerCenterAtom = (int*)tensor_numPairsPerCenterAtom.data_ptr();
  int* d_centerAtomStartIdx = (int*)tensor_centerAtomStartIdx.data_ptr();

  auto grad_force_coord_Rij = torch::zeros({nRadialRij}, coordinates_t.options().requires_grad(false));

  int block_size = 64;
  int nblocks = (nRadialRij + block_size - 1) / block_size;
  pairwiseDistance_backward_or_doublebackward<true><<<nblocks, block_size, 0, stream>>>(
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_force_coord_Rij.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_radialRij,
      nRadialRij);

  nblocks = (nRadialRij * 8 + block_size - 1) / block_size;
  cuRadialAEVs_backward_or_doublebackward<true, int, float, 8><<<nblocks, block_size, 0, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_force_coord_Rij.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      d_radialRij,
      aev_params,
      nRadialRij);

  auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;

    return (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };

  block_size = 32;
  const int nthreads_per_catom = 32;
  const int nblocks_angAEV = (ncenter_atoms * nthreads_per_catom + block_size - 1) / block_size;
  int smem_size_aligned = smem_size(maxnbrs_per_atom_aligned, block_size / nthreads_per_catom);

  cuAngularAEVs_backward_or_doublebackward<true><<<nblocks_angAEV, block_size, smem_size_aligned, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_angularRij,
      d_centralAtom,
      d_numPairsPerCenterAtom,
      d_centerAtomStartIdx,
      aev_params,
      maxnbrs_per_atom_aligned,
      angular_length_aligned,
      ncenter_atoms);

  return grad_grad_aev;
}

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
    AEVScalarParams<float> aev_params(prectx->saved_data["aev_params"]);
    c10::List<int64_t> int_list = prectx->saved_data["int_list"].toIntList();
    int total_natom_pairs = int_list[0], nRadialRij = int_list[1], nAngularRij = int_list[2];
    int maxnbrs_per_atom_aligned = int_list[3], angular_length_aligned = int_list[4];
    int ncenter_atoms = int_list[5];

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
    AEVScalarParams<float> aev_params(ctx->saved_data["aev_params"]);
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
  Result res = cuaev_forward<float>(
      coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
  return res.aev_t;
}

class CuaevAutograd : public torch::autograd::Function<CuaevAutograd> {
 public:
  static Tensor forward(AutogradContext* ctx, AEV_INPUT) {
    at::AutoNonVariableTypeMode g;
    Result res = cuaev_forward<float>(
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

TORCH_LIBRARY(cuaev, m) {
  m.def("cuComputeAEV", cuaev_cuda);
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("cuComputeAEV", cuaev_cuda);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("cuComputeAEV", cuaev_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
