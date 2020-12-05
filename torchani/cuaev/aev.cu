#include <stdio.h>
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
using torch::autograd::tensor_list;

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

  AEVScalarParams(const torch::IValue& aev_params_ivalue){
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

#define MAX_NSPECIES 10
__constant__ int csubaev_offsets[MAX_NSPECIES * MAX_NSPECIES];

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

// every block compute blocksize RIJ's gradient by column major, to avoid atomicAdd waiting
template <typename DataT, typename IndexT = int>
__global__ void pairwiseDistance_backward(
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> pos_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> grad_radial_dist,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> grad_coord,
    PairDist<DataT>* d_radialRij,
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

  DataT grad_dist_coord_x = delx / Rij;
  DataT grad_dist_coord_y = dely / Rij;
  DataT grad_dist_coord_z = delz / Rij;
  DataT grad_radial_dist_item = grad_radial_dist[gidx];

  atomicAdd(&grad_coord[mol_idx][j][0], grad_radial_dist_item * grad_dist_coord_x);
  atomicAdd(&grad_coord[mol_idx][j][1], grad_radial_dist_item * grad_dist_coord_y);
  atomicAdd(&grad_coord[mol_idx][j][2], grad_radial_dist_item * grad_dist_coord_z);
  atomicAdd(&grad_coord[mol_idx][i][0], -grad_radial_dist_item * grad_dist_coord_x);
  atomicAdd(&grad_coord[mol_idx][i][1], -grad_radial_dist_item * grad_dist_coord_y);
  atomicAdd(&grad_coord[mol_idx][i][2], -grad_radial_dist_item * grad_dist_coord_z);
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

  int threads_per_catom = TILEX * TILEY;
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

      for (int srcLane = 0; srcLane < 32 && (kk_start + srcLane) < jnum; ++srcLane) {
        int kk = kk_start + srcLane;
        DataT theta_ijk = __shfl_sync(0xFFFFFFFF, theta, srcLane);

        const DataT Rik = sdist[kk];
        SpeciesT type_k = stype[kk];

        DataT fc_ik = sfc[kk];

        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = csubaev_offsets[type_j * num_species + type_k];
        // IndexT aev_offset = aev_params.radial_length + subaev_offset;

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

  // SpeciesT type_i = species_t[mol_idx][i];
  SpeciesT type_j = species_t[mol_idx][j];

  DataT fc = 0.5 * cos(PI * Rij / aev_params.Rcr) + 0.5;

  for (int ishfr = laneIdx; ishfr < nShfR; ishfr += THREADS_PER_RIJ) {
    DataT ShfR = ShfR_t[ishfr];

    DataT GmR = 0.25 * exp(-EtaR * (Rij - ShfR) * (Rij - ShfR)) * fc;

    atomicAdd(&aev_t[mol_idx][i][type_j * aev_params.radial_sublength + ishfr], GmR);
  }
}

// every <THREADS_PER_RIJ> threads take care of 1 RIJ, and iterate <nShfR / THREADS_PER_RIJ> times
template <typename SpeciesT, typename DataT, int THREADS_PER_RIJ>
__global__ void cuRadialAEVs_backward(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> grad_radial_dist,
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

  // SpeciesT type_i = species_t[mol_idx][i];
  SpeciesT type_j = species_t[mol_idx][j];

  DataT fc = 0.5 * cos(PI * Rij / aev_params.Rcr) + 0.5;
  DataT fc_grad = -0.5 * (PI / aev_params.Rcr) * sin(PI * Rij / aev_params.Rcr);

  for (int ishfr = laneIdx; ishfr < nShfR; ishfr += THREADS_PER_RIJ) {
    DataT ShfR = ShfR_t[ishfr];

    DataT GmR = 0.25 * exp(-EtaR * (Rij - ShfR) * (Rij - ShfR));
    DataT GmR_grad = -EtaR * (-2 * ShfR + 2 * Rij) * GmR;

    DataT grad_output_item = grad_output[mol_idx][i][type_j * aev_params.radial_sublength + ishfr];
    DataT grad_radial_dist_item = grad_output_item * (GmR_grad * fc + GmR * fc_grad);

    atomicAdd(&grad_radial_dist[idx], grad_radial_dist_item);
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

void initConsts(AEVScalarParams<float>& aev_params, cudaStream_t stream) {
  int num_species = aev_params.num_species;
  assert(num_species <= MAX_NSPECIES);
  // precompute the aev offsets and load to constand memory
  int* subaev_offsets = new int[num_species * num_species];
  for (int t = 0; t < num_species; ++t) {
    int offset = 0;
    for (int s = 0; s < num_species; s++) {
      if (t < num_species - s) {
        subaev_offsets[s * num_species + s + t] = aev_params.angular_sublength * (offset + t);
        subaev_offsets[(s + t) * num_species + s] = aev_params.angular_sublength * (offset + t);
      }
      offset += num_species - s;
    }
  }
  cudaMemcpyToSymbolAsync(
      csubaev_offsets, subaev_offsets, sizeof(int) * num_species * num_species, 0, cudaMemcpyDefault, stream);
  delete[] subaev_offsets;
}

typedef struct Result {
  Tensor aev_t;
  AEVScalarParams<float> aev_params;
  Tensor tensor_Rij;
  Tensor tensor_radialRij;
  Tensor tensor_angularRij;
  int total_natom_pairs;
  int nRadialRij;
  int nAngularRij;
} Result;

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

  // precompute the aev offsets and load to constand memory
  initConsts(aev_params, stream);

  // buffer to store all the pairwise distance (Rij)
  auto total_natom_pairs = n_molecules * max_natoms_per_mol * max_natoms_per_mol;
  auto d_options = torch::dtype(torch::kUInt8).device(torch::kCUDA, coordinates_t.device().index());
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

  dim3 block(8, 8, 1);
  // Compute pairwise distance (Rij) for all atom pairs in a molecule
  // maximum 4096 atoms, which needs 49152 byte (48 kb) of shared memory
  pairwiseDistance<<<n_molecules, block, sizeof(float) * max_natoms_per_mol * 3, stream>>>(
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_Rij,
      max_natoms_per_mol);

  // Extract Rijs that is needed for RadialAEV comptuation i.e. all the Rij <=
  // Rcr
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

  auto buffer_centralAtom = allocator.allocate(sizeof(PairDist<float>) * nAngularRij);
  PairDist<float>* d_centralAtom = (PairDist<float>*)buffer_centralAtom.get();

  auto buffer_numPairsPerCenterAtom = allocator.allocate(sizeof(int) * nAngularRij);
  int* d_numPairsPerCenterAtom = (int*)buffer_numPairsPerCenterAtom.get();

  // group by center atom
  int ncenter_atoms = cubEncode(d_angularRij, d_centralAtom, d_numPairsPerCenterAtom, nAngularRij, d_count_out, stream);

  auto buffer_centerAtomStartIdx = allocator.allocate(sizeof(int) * ncenter_atoms);
  int* d_centerAtomStartIdx = (int*)buffer_centerAtomStartIdx.get();

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

    cuAngularAEVs<<<
        nblocks_angAEV,
        block_size,
        smem_size(maxnbrs_per_atom_aligned, block_size / nthreads_per_catom),
        stream>>>(
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
        align<4>(aev_params.angular_length),
        ncenter_atoms);
  }
  return {
      aev_t, aev_params, tensor_Rij, tensor_radialRij, tensor_angularRij, total_natom_pairs, nRadialRij, nAngularRij};
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
    int nAngularRij) {
  using namespace torch::indexing;
  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto grad_coord = torch::zeros(coordinates_t.sizes(), coordinates_t.options().requires_grad(false)); // [2, 5, 3]
  auto grad_output_radial = grad_output.index({Ellipsis, Slice(None, aev_params.radial_length)}); // [2, 5, 64]
  auto grad_output_angular = grad_output.index({Ellipsis, Slice(aev_params.radial_length, None)}); // [2, 5, 320]

  PairDist<float>* d_radialRij = (PairDist<float>*)tensor_radialRij.data_ptr();

  Tensor grad_radial_dist = torch::zeros(nRadialRij, coordinates_t.options().requires_grad(false));
  // float* grad_radial_dist_ptr = (float*)grad_radial_dist.data_ptr();

  int block_size = 64;
  int nblocks = (nRadialRij * 8 + block_size - 1) / block_size;
  cuRadialAEVs_backward<int, float, 8><<<nblocks, block_size, 0, stream>>>(
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
  pairwiseDistance_backward<<<nblocks, block_size, 0, stream>>>(
      coordinates_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_radial_dist.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      d_radialRij,
      nRadialRij);

  return grad_coord;
}

#define AEV_INPUT                                                                                                     \
  const Tensor& coordinates_t, const Tensor& species_t, double Rcr_, double Rca_, const Tensor& EtaR_t, const Tensor& ShfR_t, const Tensor& EtaA_t, \
      const Tensor& Zeta_t, const Tensor& ShfA_t, const Tensor& ShfZ_t, int64_t num_species_

Tensor cuaev_cuda(AEV_INPUT) {
  printf("calling cuda \n");
  Result res = cuaev_forward<float>(
      coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
  return res.aev_t;
}

class CuaevAutograd : public torch::autograd::Function<CuaevAutograd> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, AEV_INPUT) {
    at::AutoNonVariableTypeMode g;
    printf("calling autograd::forward \n");
    Result res = cuaev_forward<float>(
        coordinates_t, species_t, Rcr_, Rca_, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species_);
    if (coordinates_t.requires_grad()) {
      printf("saving context \n");
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
                              ShfZ_t});
      ctx->saved_data["aev_params"] = static_cast<torch::IValue>(res.aev_params);
      ctx->saved_data["int_list"] = c10::List<int64_t> {res.total_natom_pairs, res.nRadialRij, res.nAngularRij};
    }
    return res.aev_t;
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
    printf("calling autograd::backward \n");
    auto saved = ctx->get_saved_variables();
    auto coordinates_t = saved[0], species_t = saved[1];
    auto tensor_Rij = saved[2], tensor_radialRij = saved[3], tensor_angularRij = saved[4];
    auto EtaR_t = saved[5], ShfR_t = saved[6], EtaA_t = saved[7], Zeta_t = saved[8], ShfA_t = saved[9],
         ShfZ_t = saved[10];
    AEVScalarParams<float> aev_params (ctx->saved_data["aev_params"]);
    c10::List<int64_t> int_list = ctx->saved_data["int_list"].toIntList();
    int total_natom_pairs = (int)int_list[0];
    int nRadialRij = (int)int_list[1];
    int nAngularRij = (int)int_list[2];

    Tensor grad_coord = cuaev_backward(
        grad_outputs[0],
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
        nAngularRij);

    return {
        grad_coord, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

Tensor cuaev_autograd(AEV_INPUT) {
  printf("calling autograd \n");
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
