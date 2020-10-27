#include "nvToolsExt.h"
#include <chrono>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <torch/extension.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <THC/THC.h>
#include <ATen/Context.h>
#include <THC/THCThrustAllocator.cuh>

#define PI 3.141592653589793

template <typename DataT, typename IndexT = int> struct AEVScalarParams {
  DataT Rcr;
  DataT Rca;

  IndexT radial_sublength;
  IndexT radial_length;
  IndexT angular_sublength;
  IndexT angular_length;
  IndexT num_species;
};

// cub::CachingDeviceAllocator
//     g_allocator(2, 10, cub::CachingDeviceAllocator::INVALID_BIN, 1e9, false,
//                 false);

#define MAX_NSPECIES 10
__constant__ int csubaev_offsets[MAX_NSPECIES * MAX_NSPECIES];

template <typename DataT> struct PairDist {
  DataT Rij;
  int midx;
  short i;
  short j;
};

// used to group Rijs by atom id
template <typename DataT>
__host__ __device__ bool operator==(const PairDist<DataT> &lhs,
                                    const PairDist<DataT> &rhs) {
  return lhs.midx == rhs.midx && lhs.i == rhs.i;
}

/// Alignment of memory. Must be a power of two
/// \tparam boundary Boundary to align to (NOTE: must be power of 2)
/// \param value Input value that is to be aligned
/// \return Value aligned to boundary
template <int32_t boundary>
__host__ __device__ __forceinline__ int align(const int &value) {
  static_assert((boundary & (boundary - 1)) == 0,
                "Boundary for align must be power of 2");
  return (value + boundary) & ~(boundary - 1);
}

template <typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits>
        species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    PairDist<DataT> *d_Rij, IndexT max_natoms_per_mol) {

  extern __shared__ DataT spos[];
  DataT *sx = &spos[0];
  DataT *sy = &spos[max_natoms_per_mol];
  DataT *sz = &spos[2 * max_natoms_per_mol];

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

template <typename SpeciesT, typename DataT, typename IndexT = int,
          int TILEX = 8, int TILEY = 4>
__global__ void cuAngularAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits>
        species_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    PairDist<DataT> *d_Rij, PairDist<DataT> *d_centralAtom,
    int *d_nPairsPerCenterAtom, int *d_centerAtomStartIdx,
    AEVScalarParams<DataT, IndexT> aev_params, int maxnbrs_per_atom_aligned,
    int angular_length_aligned, int ncentral_atoms) {
  extern __shared__ DataT smem[];

  int threads_per_catom = TILEX * TILEY;
  int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int cIdx = gIdx / threads_per_catom; // central atom id

  if (cIdx >= ncentral_atoms)
    return;

  int groupIdx = threadIdx.x / threads_per_catom;
  int laneIdx = threadIdx.x % threads_per_catom;
  int ncatom_per_tpb = blockDim.x / threads_per_catom;

  DataT *saev = &smem[groupIdx * angular_length_aligned];

  int offset = ncatom_per_tpb * angular_length_aligned;
  DataT *sdx = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT *sdy = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT *sdz = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT *sdist = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  DataT *sfc = &smem[offset + groupIdx * maxnbrs_per_atom_aligned];

  offset += ncatom_per_tpb * maxnbrs_per_atom_aligned;
  int *stype = (int *)&smem[offset + groupIdx * maxnbrs_per_atom_aligned];

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

  for (int iaev = laneIdx; iaev < aev_params.angular_length;
       iaev += threads_per_catom) {
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

  for (int jj = 0; jj < jnum; jj++) {
    const DataT Rij = sdist[jj];
    SpeciesT type_j = stype[jj];

    DataT fc_ij = sfc[jj];

    for (int kk_start = jj + 1; kk_start < jnum;
         kk_start += threads_per_catom) {

      int kk = kk_start + laneIdx;
      DataT theta = 0;
      if (kk < jnum) {
        const DataT Rik = sdist[kk];
        theta = acos(
            0.95 * (sdx[jj] * sdx[kk] + sdy[jj] * sdy[kk] + sdz[jj] * sdz[kk]) /
            (Rij * Rik));
      }

      for (int srcLane = 0; kk_start + srcLane < min(32, jnum); ++srcLane) {
        int kk = kk_start + srcLane;
        DataT theta_ijk = __shfl_sync(0xFFFFFFFF, theta, srcLane);

        const DataT Rik = sdist[kk];
        SpeciesT type_k = stype[kk];

        DataT fc_ik = sfc[kk];

        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = csubaev_offsets[type_j * num_species + type_k];
        IndexT aev_offset = aev_params.radial_length + subaev_offset;

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

  for (int iaev = laneIdx; iaev < aev_params.angular_length;
       iaev += threads_per_catom) {
    aev_t[mol_idx][i][aev_params.radial_length + iaev] = saev[iaev];
  }
}

template <typename SpeciesT, typename DataT, int THREADS_PER_RIJ>
__global__ void cuRadialAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits>
        species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    PairDist<DataT> *d_Rij, AEVScalarParams<DataT, int> aev_params,
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

  SpeciesT type_i = species_t[mol_idx][i];
  SpeciesT type_j = species_t[mol_idx][j];

  DataT fc = 0.5 * cos(PI * Rij / aev_params.Rcr) + 0.5;

  for (int ishfr = laneIdx; ishfr < nShfR; ishfr += THREADS_PER_RIJ) {
    DataT ShfR = ShfR_t[ishfr];

    DataT GmR = 0.25 * exp(-EtaR * (Rij - ShfR) * (Rij - ShfR)) * fc;

    atomicAdd(&aev_t[mol_idx][i][type_j * aev_params.radial_sublength + ishfr],
              GmR);
  }
}

template <typename DataT>
void cubScan(const DataT *d_in, DataT *d_out, int num_items) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                num_items, stream);

  // Allocate temporary storage
  auto buffer_ = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_.get();
  // CubDebugExit(
  //     g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                num_items, stream);

  // CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  // cudaFree(d_temp_storage);
}

template <typename DataT, typename IndexT>
int cubEncode(const DataT *d_in, DataT *d_unique_out, IndexT *d_counts_out,
              int num_items, int *d_num_runs_out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in,
                                     d_unique_out, d_counts_out, d_num_runs_out,
                                     num_items, stream);

  // Allocate temporary storage
  auto buffer_ = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_.get();
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CubDebugExit(
  //     g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));

  // Run encoding
  cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in,
                                     d_unique_out, d_counts_out, d_num_runs_out,
                                     num_items, stream);

  int num_selected = 0;
  cudaMemcpy(&num_selected, d_num_runs_out, sizeof(int), cudaMemcpyDefault);

  // CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  // cudaFree(d_temp_storage);
  return num_selected;
}

template <typename DataT, typename LambdaOpT>
int cubDeviceSelect(const DataT *d_in, DataT *d_out, int num_items,
                    int *d_num_selected_out, LambdaOpT select_op) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                        d_num_selected_out, num_items, select_op);

  // Allocate temporary storage
  auto buffer_ = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_.get();
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // CubDebugExit(
  //     g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));

  // Run selection
  cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                        d_num_selected_out, num_items, select_op, stream);

  int num_selected = 0;
  cudaMemcpy(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDefault);

  // CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  // cudaFree(d_temp_storage);
  return num_selected;
}

template <typename DataT>
DataT cubMax(const DataT *d_in, int num_items, DataT *d_out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out,
                         num_items, stream);

  // Allocate temporary storage
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);
  auto buffer_ = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_.get();
  // CubDebugExit(
  //     g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));

  // Run min-reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out,
                         num_items, stream);

  int maxVal = 0;
  cudaMemcpy(&maxVal, d_out, sizeof(DataT), cudaMemcpyDefault);

  // CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  // cudaFree(d_temp_storage);
  return maxVal;
}

void initConsts(AEVScalarParams<float> &aev_params) {
  int num_species = aev_params.num_species;
  assert(num_species <= MAX_NSPECIES);
  // precompute the aev offsets and load to constand memory
  int *subaev_offsets = new int[num_species * num_species];
  for (int t = 0; t < num_species; ++t) {
    int offset = 0;
    for (int s = 0; s < num_species; s++) {
      if (t < num_species - s) {
        subaev_offsets[s * num_species + s + t] =
            aev_params.angular_sublength * (offset + t);
        subaev_offsets[(s + t) * num_species + s] =
            aev_params.angular_sublength * (offset + t);
      }
      offset += num_species - s;
    }
  }
  cudaMemcpyToSymbol(csubaev_offsets, subaev_offsets,
                     sizeof(int) * num_species * num_species);
  delete[] subaev_offsets;
}

// NOTE: assumes size of EtaA_t = Zeta_t = EtaR_t = 1
template <typename ScalarRealT = float>
void cuComputeAEV(torch::Tensor coordinates_t, torch::Tensor species_t,
                  ScalarRealT Rcr, ScalarRealT Rca, torch::Tensor EtaR_t,
                  torch::Tensor ShfR_t, torch::Tensor EtaA_t,
                  torch::Tensor Zeta_t, torch::Tensor ShfA_t,
                  torch::Tensor ShfZ_t, torch::Tensor aev_t, int num_species) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto thrust_allocator = THCThrustAllocator(at::globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(thrust_allocator).on(stream);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  
  const int n_molecules = species_t.size(0);
  const int max_natoms_per_mol = species_t.size(1);
  // std::cout << "Running cuComputeAEV with " << n_molecules << "x" <<
  // max_natoms_per_mol << "\n";

  AEVScalarParams<float> aev_params;
  aev_params.Rca = Rca;
  aev_params.Rcr = Rcr;
  aev_params.num_species = num_species;

  aev_params.radial_sublength = EtaR_t.size(0) * ShfR_t.size(0);
  aev_params.radial_length = aev_params.radial_sublength * num_species;

  aev_params.angular_sublength =
      EtaA_t.size(0) * Zeta_t.size(0) * ShfA_t.size(0) * ShfZ_t.size(0);
  aev_params.angular_length =
      aev_params.angular_sublength * (num_species * (num_species + 1) / 2);

  if (EtaR_t.size(0) != 1 || EtaA_t.size(0) != 1 || Zeta_t.size(0) != 1) {
    std::cerr << "cuda extension is currently not supported for the specified "
                 "configuration\n";
    exit(1);
  }

  // precompute the aev offsets and load to constand memory
  initConsts(aev_params);

  // buffer to store all the pairwise distance (Rij)
  PairDist<float> *d_Rij = 0;
  auto total_natom_pairs =
      n_molecules * max_natoms_per_mol * max_natoms_per_mol;
  auto buffer_size = sizeof(PairDist<float>) * total_natom_pairs;
  auto buffer_ = allocator.allocate(buffer_size);
  d_Rij = (PairDist<float> *)buffer_.get();
  // CubDebugExit(g_allocator.DeviceAllocate(
  //     (void **)&d_Rij, sizeof(PairDist<float>) * total_natom_pairs));
  // cudaMalloc((void **)&d_Rij, sizeof(PairDist<float>) * total_natom_pairs);

  // init all Rij to inf
  PairDist<float> init;
  init.Rij = std::numeric_limits<float>::infinity();
  thrust::fill(policy, d_Rij, d_Rij + total_natom_pairs, init);

  // buffer to store all the pairwise distance that is needed for Radial AEV
  // computation
  PairDist<float> *d_radialRij = 0;
  auto buffer2_ = allocator.allocate(buffer_size);
  d_radialRij = (PairDist<float> *)buffer2_.get();
  // CubDebugExit(g_allocator.DeviceAllocate(
  //     (void **)&d_radialRij, sizeof(PairDist<float>) * total_natom_pairs));
  // cudaMalloc((void **)&d_radialRij, sizeof(PairDist<float>) *
  // total_natom_pairs);

  int *d_count_out = 0;
  auto buffer3_ = allocator.allocate(sizeof(int));
  d_count_out = (int *)buffer3_.get();
  // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_count_out, sizeof(int)));
  // cudaMalloc((void **)&d_count_out, workspace_size);

  const int block_size = 64;
  auto start = std::chrono::steady_clock::now();
  if ((species_t.dtype() == torch::kInt32) &&
      (coordinates_t.dtype() == torch::kFloat32)) {

    dim3 block(8, 8, 1);
    // Compute pairwise distance (Rij) for all atom pairs in a molecule
    pairwiseDistance<<<n_molecules, block,
                       sizeof(float) * max_natoms_per_mol * 3, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_Rij, max_natoms_per_mol);

    // Extract Rijs that is needed for RadialAEV comptuation i.e. all the Rij <=
    // Rcr
    int nRadialRij = cubDeviceSelect(
        d_Rij, d_radialRij, total_natom_pairs, d_count_out,
        [=] __device__(const PairDist<float> d) { return d.Rij <= Rcr; });

    int nblocks = (nRadialRij * 8 + block_size - 1) / block_size;
    cuRadialAEVs<int, float, 8><<<nblocks, block_size, 0, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        d_radialRij, aev_params, nRadialRij);

    // reuse buffer allocated for all Rij
    // d_angularRij will store all the Rij required in Angular AEV computation
    PairDist<float> *d_angularRij = d_Rij;

    // Extract Rijs that is needed for AngularAEV comptuation i.e. all the Rij
    // <= Rca
    int nAngularRij = cubDeviceSelect(
        d_radialRij, d_angularRij, nRadialRij, d_count_out,
        [=] __device__(const PairDist<float> d) { return d.Rij <= Rca; });

    PairDist<float> *d_centralAtom = 0;
    auto buffer_size4 = sizeof(PairDist<float>) * nAngularRij;
    auto buffer4_ = allocator.allocate(buffer_size4);
    d_centralAtom = (PairDist<float> *)buffer4_.get();
    // CubDebugExit(g_allocator.DeviceAllocate(
    //     (void **)&d_centralAtom, sizeof(PairDist<float>) * nAngularRij));
    // cudaMalloc((void **)&d_centralAtom, sizeof(PairDist<float>) *
    // nAngularRij);

    int *d_numPairsPerCenterAtom = 0;
    auto buffer_size5 = sizeof(int) * nAngularRij;
    auto buffer5_ = allocator.allocate(buffer_size5);
    d_numPairsPerCenterAtom = (int *)buffer5_.get();
    // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_numPairsPerCenterAtom,
    //                                         sizeof(int) * nAngularRij));
    // cudaMalloc((void **)&d_numPairsPerCenterAtom, sizeof(int) * nAngularRij);

    // group by center atom
    int ncenter_atoms =
        cubEncode(d_angularRij, d_centralAtom, d_numPairsPerCenterAtom,
                  nAngularRij, d_count_out);

    int *d_centerAtomStartIdx = 0;
    auto buffer_size6 = sizeof(int) * ncenter_atoms;
    auto buffer6_ = allocator.allocate(buffer_size6);
    d_centerAtomStartIdx = (int *)buffer6_.get();
    // CubDebugExit(g_allocator.DeviceAllocate((void **)&d_centerAtomStartIdx,
    //                                         sizeof(int) * ncenter_atoms));
    // cudaMalloc((void **)&d_centerAtomStartIdx, sizeof(int) * ncenter_atoms);

    cubScan(d_numPairsPerCenterAtom, d_centerAtomStartIdx, ncenter_atoms);
    {

      // ncenter_atoms = 1;
      const int nthreads_per_catom = 32;
      const int nblocks_angAEV =
          (ncenter_atoms * nthreads_per_catom + block_size - 1) / block_size;
      auto smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
        int sm_aev = sizeof(float) * align<4>(aev_params.angular_length);
        int sxyz = sizeof(float) * max_nbrs * 3;
        int sRij = sizeof(float) * max_nbrs;
        int sfc = sizeof(float) * max_nbrs;
        int sj = sizeof(int) * max_nbrs;

        return (sm_aev + sxyz + sRij + sfc + sj) * ncatom_per_tpb;
      };

      int maxNbrsPerCenterAtom =
          cubMax(d_numPairsPerCenterAtom, ncenter_atoms, d_count_out);

      int maxnbrs_per_atom_aligned = align<4>(maxNbrsPerCenterAtom);

      cuAngularAEVs<<<nblocks_angAEV, block_size,
                      smem_size(maxnbrs_per_atom_aligned,
                                block_size / nthreads_per_catom), stream>>>(
          species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
          coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          d_angularRij, d_centralAtom, d_numPairsPerCenterAtom,
          d_centerAtomStartIdx, aev_params, maxnbrs_per_atom_aligned,
          align<4>(aev_params.angular_length), ncenter_atoms);
    }

    // CubDebugExit(g_allocator.DeviceFree(d_centerAtomStartIdx));
    // CubDebugExit(g_allocator.DeviceFree(d_numPairsPerCenterAtom));
    // CubDebugExit(g_allocator.DeviceFree(d_centralAtom));
    // CubDebugExit(g_allocator.DeviceFree(d_count_out));
    // CubDebugExit(g_allocator.DeviceFree(d_radialRij));
    // CubDebugExit(g_allocator.DeviceFree(d_Rij));
    // cudaFree(d_numPairsPerCenterAtom);
    // cudaFree(d_centralAtom);
    // cudaFree(d_count_out);
    // cudaFree(d_radialRij);
    // cudaFree(d_Rij);
  } else {
    std::cerr << "Type Error!\n";
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuComputeAEV", &cuComputeAEV<float>, "CUDA method to compute AEVs");
}
