#include <aev.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuaev_cub.cuh>

#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>

#define PI 3.141592653589793
#define MAX_NUMJ_PER_I_IN_RCR 1000 // normally this value is less than 100 when Rcr is 5 A
#define SMOOTH_CUTOFF_ORDER 2
#define SMOOTH_CUTOFF_EPS 1e-10

using torch::Tensor;

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

// Convert pair index to atom j and k indices,
// the orders of j and k indices do not matter
//
// e.g. jnum is 4, there are totally (4 * 3 / 2) = 6 three-body pairs,
// so this function convert following indices n
// [ 0,  1,  2,  3,  4,  5]
// to k:
// [ 1,  2,  2,  3,  3,  3]
// then j will be:
// [ 0,  0,  1,  0,  1,  2]
__device__ __forceinline__ int2 pairidx_to_jk(int n) {
  int kk = ceil((sqrt(8 * (n + 1) + 1.f) - 1) / 2.f); // x (x + 1) / 2 = n --> x = (-1 + sqrt(1 + 8n)) / 2
  int jj = n - kk * (kk - 1) / 2;
  return make_int2(jj, kk);
}

template <typename DataT>
__device__ __forceinline__ DataT cosine_cutoff_fwd(DataT Rij, DataT Rc) {
  return 0.5f * __cosf(PI * Rij / Rc) + 0.5f;
}

template <typename DataT>
__device__ __forceinline__ DataT cosine_cutoff_bwd(DataT Rij, DataT Rc) {
  return -0.5f * (PI / Rc) * __sinf(PI * Rij / Rc);
}

template <typename DataT>
__device__ __forceinline__ DataT smooth_cutoff_fwd(DataT Rij, DataT Rc) {
  DataT eps = SMOOTH_CUTOFF_EPS;
#if (SMOOTH_CUTOFF_ORDER == 2)
  DataT p = (Rij / Rc) * (Rij / Rc);
#elif (SMOOTH_CUTOFF_ORDER == 3)
  DataT p = (Rij / Rc) * (Rij / Rc) * (Rij / Rc);
#else
  DataT p = __powf(Rij / Rc, SMOOTH_CUTOFF_ORDER);
#endif
  DataT m = std::max(eps, 1 - p);
  return __expf(1 - 1 / m);
}

template <typename DataT>
__device__ __forceinline__ DataT smooth_cutoff_bwd(DataT Rij, DataT Rc) {
  DataT eps = SMOOTH_CUTOFF_EPS;
  int order = SMOOTH_CUTOFF_ORDER;
#if (SMOOTH_CUTOFF_ORDER == 2)
  DataT p = (Rij / Rc) * (Rij / Rc);
#elif (SMOOTH_CUTOFF_ORDER == 3)
  DataT p = (Rij / Rc) * (Rij / Rc) * (Rij / Rc);
#else
  DataT p = __powf(Rij / Rc, order);
#endif
  DataT m = std::max(eps, 1 - p);
  DataT step_fn = (-eps - p + 1) >= 0 ? 1 : 0;
  return -step_fn * order * p * __expf(1 - 1 / m) / (Rij * m * m);
}

template <typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistance(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const DataT* pos_p,
    torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> radialNumPairsPerAtom_t,
    AtomI* __restrict__ atom_i,
    int* __restrict__ atomJ_p,
    float* __restrict__ distJ_p,
    const DataT Rcr,
    const IndexT max_natoms_per_mol,
    const IndexT max_numj_per_i_in_Rcr) {
  extern __shared__ float smem[];
  int* s_pcounter_i = reinterpret_cast<int*>(&smem[0]);
  int* s_species = reinterpret_cast<int*>(&smem[max_natoms_per_mol]);
  float3* s_pos = reinterpret_cast<float3*>(&smem[max_natoms_per_mol * 2]);

  const float3* pos_p_3 = reinterpret_cast<const float3*>(pos_p);

  int mol_idx = blockIdx.x;
  int tIdx = blockDim.x * threadIdx.y + threadIdx.x;

  for (int i = tIdx; i < max_natoms_per_mol; i += blockDim.x * blockDim.y) {
    SpeciesT specie_i = species_t[mol_idx][i];
    s_species[i] = specie_i;
    s_pcounter_i[i] = 0;
    if (specie_i != -1) {
      s_pos[i] = pos_p_3[max_natoms_per_mol * mol_idx + i];
    }
  }
  __syncthreads();

  int pairs_per_mol = max_natoms_per_mol * max_numj_per_i_in_Rcr;

  for (int i = threadIdx.y; i < max_natoms_per_mol; i += blockDim.y) {
    SpeciesT specie_i = s_species[i];
    if (specie_i != -1) {
      float3 pos_i = s_pos[i];

      for (int j = threadIdx.x + i + 1; j < max_natoms_per_mol; j += blockDim.x) {
        SpeciesT specie_j = s_species[j];

        if (specie_j != -1) {
          float3 delta = make_float3(s_pos[j].x - pos_i.x, s_pos[j].y - pos_i.y, s_pos[j].z - pos_i.z);
          DataT Rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
          DataT Rij = sqrt(Rsq);
          if (Rij <= Rcr) {
            // for atom i
            int pidx_i = atomicAdd(&s_pcounter_i[i], 1);
            CUDA_KERNEL_ASSERT(pidx_i < max_numj_per_i_in_Rcr); // check to avoid illegal memory access
            atomJ_p[mol_idx * pairs_per_mol + i * max_numj_per_i_in_Rcr + pidx_i] = j;
            distJ_p[mol_idx * pairs_per_mol + i * max_numj_per_i_in_Rcr + pidx_i] = Rij;
            // for atom j
            int pidx_j = atomicAdd(&s_pcounter_i[j], 1);
            CUDA_KERNEL_ASSERT(pidx_j < max_numj_per_i_in_Rcr);
            atomJ_p[mol_idx * pairs_per_mol + j * max_numj_per_i_in_Rcr + pidx_j] = i;
            distJ_p[mol_idx * pairs_per_mol + j * max_numj_per_i_in_Rcr + pidx_j] = Rij;
          } // if Rij is within Rcr
        } // if j is not padding atom and i is not j
      } // loop over atom j

    } // if i is not padding atom
    atom_i[mol_idx * max_natoms_per_mol + i] = {mol_idx, i};
  }
  __syncthreads();

  for (int i = tIdx; i < max_natoms_per_mol; i += blockDim.x * blockDim.y) {
    radialNumPairsPerAtom_t[mol_idx * max_natoms_per_mol + i] = s_pcounter_i[i];
  }
}

// Every row in the block compute all neighbors J for 1 atomI.
template <int ATOM_I_PER_BLOCK, int ATOM_J_PER_TILE, typename SpeciesT, typename DataT, typename IndexT = int>
__global__ void pairwiseDistanceSingleMolecule(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const DataT* pos_p,
    torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> radialNumPairsPerAtom_t,
    AtomI* __restrict__ atom_i,
    int* __restrict__ atomJ_p,
    float* __restrict__ distJ_p,
    const DataT Rcr,
    const IndexT max_natoms_per_mol,
    const IndexT max_numj_per_i_in_Rcr) {
  __shared__ int s_pcounter_i[ATOM_I_PER_BLOCK];
  constexpr int BLOCK_SIZE = ATOM_I_PER_BLOCK * ATOM_J_PER_TILE;
  __shared__ float3 s_coord_j[BLOCK_SIZE];
  const float3* pos_p_3 = reinterpret_cast<const float3*>(pos_p);

  constexpr int mol_idx = 0;
  int pairs_per_mol = max_natoms_per_mol * max_numj_per_i_in_Rcr;
  int i = blockIdx.x * blockDim.y + threadIdx.y;
  int ii = threadIdx.y;
  int tIdx = blockDim.x * threadIdx.y + threadIdx.x;
  int num_tiles = (max_natoms_per_mol + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // i >= max_natoms_per_mol is still needed to load share memory for j
  SpeciesT specie_i;
  float3 coord_i;
  if (i < max_natoms_per_mol) {
    specie_i = species_t[mol_idx][i];
    coord_i = pos_p_3[mol_idx * max_natoms_per_mol + i];
  }

  if (tIdx < ATOM_I_PER_BLOCK)
    s_pcounter_i[tIdx] = 0;

  for (int tileidx = 0; tileidx < num_tiles; tileidx++) {
    // load 1 block size of atoms j into share memory
    int jidx = BLOCK_SIZE * tileidx + tIdx;
    if (jidx < max_natoms_per_mol) {
      s_coord_j[tIdx] = pos_p_3[max_natoms_per_mol * mol_idx + jidx];
    }

    __syncthreads();
    for (int jj = threadIdx.x; jj < BLOCK_SIZE && i < max_natoms_per_mol; jj += blockDim.x) {
      int j = jj + BLOCK_SIZE * tileidx;
      if (j < max_natoms_per_mol) {
        float3 delta =
            make_float3(s_coord_j[jj].x - coord_i.x, s_coord_j[jj].y - coord_i.y, s_coord_j[jj].z - coord_i.z);
        SpeciesT specie_j = species_t[mol_idx][j];
        DataT Rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (specie_i != -1 && specie_j != -1 && i != j) {
          DataT Rij = sqrt(Rsq);
          if (Rij <= Rcr) {
            int pidx = atomicAdd(&s_pcounter_i[ii], 1);
            CUDA_KERNEL_ASSERT(pidx < max_numj_per_i_in_Rcr);
            atomJ_p[mol_idx * pairs_per_mol + i * max_numj_per_i_in_Rcr + pidx] = j;
            distJ_p[mol_idx * pairs_per_mol + i * max_numj_per_i_in_Rcr + pidx] = Rij;
          }
        }
      }
    }
    __syncthreads();
  }

  i = tIdx + blockIdx.x * blockDim.y;
  if (tIdx < ATOM_I_PER_BLOCK && i < max_natoms_per_mol) {
    radialNumPairsPerAtom_t[i] = s_pcounter_i[tIdx];
    atom_i[mol_idx * max_natoms_per_mol + i] = {mol_idx, i};
  }
}

template <
    int BLOCK_X,
    int BLOCK_Y,
    bool use_cos_cutoff,
    typename SpeciesT,
    typename DataT,
    typename IndexT = int,
    int TILE_SIZE = 4,
    int TILE_PER_WARP = 8>
__global__ void cuAngularAEVs(
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const DataT* pos_p,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    const int* __restrict__ atomJ,
    const float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ numJPerI,
    const int* __restrict__ startIdxJ,
    float Rca,
    int angular_length,
    int angular_sublength,
    int radial_length,
    int num_species,
    int maxnbrs_per_atom,
    int ncentral_atoms) {
  constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y;

  extern __shared__ DataT smem[];
  __shared__ float s_theta[BLOCK_SIZE];

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx
  const float3* pos_p_3 = reinterpret_cast<const float3*>(pos_p);
  const int max_natoms_per_mol = species_t.size(1);
  int jnum = numJPerI[cIdx];

  if (cIdx >= ncentral_atoms)
    return;
  if (jnum < 2)
    return;

  int laneIdx = threadIdx.x;

  DataT* saev = &smem[0];

  int offset = angular_length;
  float3* svec = reinterpret_cast<float3*>(&smem[offset]);

  offset += 3 * maxnbrs_per_atom;
  DataT* sdist = &smem[offset];

  offset += maxnbrs_per_atom;
  DataT* sfc = &smem[offset];

  offset += maxnbrs_per_atom;
  int* s_species = (int*)&smem[offset];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);

  int start_idx = startIdxJ[cIdx];
  int totalpairs = jnum * (jnum - 1) / 2;
  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
    saev[iaev] = 0;
  }

  float3 coord_i = pos_p_3[mol_idx * max_natoms_per_mol + i];

  for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT specie_j = species_t[mol_idx][j];
    float3 coord_j = pos_p_3[mol_idx * max_natoms_per_mol + j];
    svec[jj] = make_float3(coord_j.x - coord_i.x, coord_j.y - coord_i.y, coord_j.z - coord_i.z);
    s_species[jj] = specie_j;
    sdist[jj] = Rij;
    if (use_cos_cutoff)
      sfc[jj] = cosine_cutoff_fwd(Rij, Rca);
    else
      sfc[jj] = smooth_cutoff_fwd(Rij, Rca);
  }
  __syncthreads();

  short2 tile = make_short2(laneIdx % TILE_SIZE, laneIdx / TILE_SIZE);
  int num_pre_process = (totalpairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // cache aev result on smem
  for (int ppIdx = 0; ppIdx < num_pre_process; ppIdx++) {
    // store 1 blocksize of theta to share mem
    int m = tIdx + ppIdx * BLOCK_SIZE;
    if (m < totalpairs) {
      int2 jk = pairidx_to_jk(m);
      int jj = jk.x, kk = jk.y;
      const DataT Rij = sdist[jj];
      const DataT Rik = sdist[kk];
      s_theta[tIdx] =
          acosf(0.95f * (svec[jj].x * svec[kk].x + svec[jj].y * svec[kk].y + svec[jj].z * svec[kk].z) / (Rij * Rik));
    }
    __syncthreads();
    // run angular calculation
    for (int nn = threadIdx.y * TILE_PER_WARP + tile.y; nn < BLOCK_SIZE; nn += BLOCK_Y * TILE_PER_WARP) {
      int n = nn + ppIdx * BLOCK_SIZE;
      if (n < totalpairs) {
        int2 jk = pairidx_to_jk(n);
        int jj = jk.x, kk = jk.y;
        const DataT Rij = sdist[jj];
        SpeciesT specie_j = s_species[jj];
        DataT fc_ij = sfc[jj];
        const DataT Rik = sdist[kk];
        SpeciesT specie_k = s_species[kk];
        DataT fc_ik = sfc[kk];

        DataT theta = s_theta[nn];
        DataT Rijk = (Rij + Rik) / 2;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = angular_sublength * csubaev_offsets(specie_j, specie_k, num_species);

        for (int ishfr = tile.x; ishfr < nShfA; ishfr += TILE_SIZE) {
          DataT ShfA = __ldg(&ShfA_t[ishfr]);
          DataT factor2 = __expf(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));

          for (int itheta = 0; itheta < nShfZ; itheta++) {
            DataT ShfZ = __ldg(&ShfZ_t[itheta]);
            DataT factor1 = __powf((1 + __cosf(theta - ShfZ)) / 2, Zeta);

            DataT res = 2 * factor1 * factor2 * fc_ijk;

            atomicAdd(&saev[subaev_offset + ishfr * nShfZ + itheta], res);
          }
        }
      }
    }
    __syncthreads();
  }

  // write aev result to global memory
  for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
    aev_t[mol_idx][i][radial_length + iaev] = saev[iaev];
  }
}

template <
    bool is_double_backward,
    int BLOCK_X,
    int BLOCK_Y,
    bool use_cos_cutoff,
    typename SpeciesT,
    typename DataT,
    typename IndexT = int,
    int TILE_SIZE = 4,
    int TILE_PER_WARP = 8>
__global__ void cuAngularAEVs_backward_or_doublebackward(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const DataT* pos_p,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfZ_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaA_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> Zeta_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_output, // for backward, this is daev, for double backward, this is dforce (i.e. ddcoord)
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_input, // for backward, this is dcoord, for double backward, this is ddaev
    const torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> atomJ,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> distJ,
    const AtomI* __restrict__ atom_i,
    int* numJPerI,
    int* startIdxJ,
    float Rca,
    int angular_length,
    int angular_sublength,
    int radial_length,
    int num_species,
    int maxnbrs_per_atom,
    int ncentral_atoms) {
  constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y;

  extern __shared__ DataT smem[];
  __shared__ float s_theta[BLOCK_SIZE];
  __shared__ float s_vij_vik_dot[BLOCK_SIZE];
  __shared__ float s_vij_factor[BLOCK_SIZE];
  __shared__ float s_vik_factor[BLOCK_SIZE];

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx
  const float3* pos_p_3 = reinterpret_cast<const float3*>(pos_p);
  const float3* grad_force_3 = reinterpret_cast<const float3*>(&grad_output[0][0][0]); // only for double backward
  const int max_natoms_per_mol = species_t.size(1);
  int jnum = numJPerI[cIdx];

  if (cIdx >= ncentral_atoms)
    return;
  if (jnum < 2)
    return;

  int laneIdx = threadIdx.x;

  // for backward, reading daev once to share mem to minimize bandwidth
  // for double backward, save the output of ddaev, and do atomicadd on saev to minimize bandwidth
  DataT* saev = &smem[0];

  int offset = angular_length;
  float3* svec = reinterpret_cast<float3*>(&smem[offset]);

  offset += 3 * maxnbrs_per_atom;
  float3* spos_j_grad = reinterpret_cast<float3*>(&smem[offset]); // only for backward

  offset += 3 * maxnbrs_per_atom;
  DataT* sdist = &smem[offset];

  offset += maxnbrs_per_atom;
  DataT* sfc = &smem[offset];

  offset += maxnbrs_per_atom;
  DataT* sfc_grad = &smem[offset];

  offset += maxnbrs_per_atom;
  int* s_species = (int*)&smem[offset];

  DataT EtaA = EtaA_t[0];
  DataT Zeta = Zeta_t[0];

  IndexT nShfA = ShfA_t.size(0);
  IndexT nShfZ = ShfZ_t.size(0);

  int start_idx = startIdxJ[cIdx];
  int totalpairs = jnum * (jnum - 1) / 2;
  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  float3 coord_i = pos_p_3[mol_idx * max_natoms_per_mol + i];
  float3 spos_i_grad;
  float3 grad_force_i;

  // initlize share memories
  if (is_double_backward) {
    for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
      saev[iaev] = 0.0f;
    }
    grad_force_i = grad_force_3[mol_idx * max_natoms_per_mol + i];
  } else {
    for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
      saev[iaev] = grad_output[mol_idx][i][radial_length + iaev];
    }
    spos_i_grad = make_float3(0.f, 0.f, 0.f);
  }

  for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT specie_j = species_t[mol_idx][j];
    float3 coord_j = pos_p_3[mol_idx * max_natoms_per_mol + j];
    svec[jj] = make_float3(coord_j.x - coord_i.x, coord_j.y - coord_i.y, coord_j.z - coord_i.z);
    s_species[jj] = specie_j;
    sdist[jj] = Rij;
    if (use_cos_cutoff) {
      sfc[jj] = cosine_cutoff_fwd(Rij, Rca);
      sfc_grad[jj] = cosine_cutoff_bwd(Rij, Rca);
    } else {
      sfc[jj] = smooth_cutoff_fwd(Rij, Rca);
      sfc_grad[jj] = smooth_cutoff_bwd(Rij, Rca);
    }
    spos_j_grad[jj] = make_float3(0.f, 0.f, 0.f);
  }
  __syncthreads();

  short2 tile = make_short2(laneIdx % TILE_SIZE, laneIdx / TILE_SIZE);
  const DataT tc = 0.95f; // theta constant factor
  int num_pre_process = (totalpairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int ppIdx = 0; ppIdx < num_pre_process; ppIdx++) {
    // preprocess 1 blocksize of theta, vij_vik_dot, vij_factor and vik_factor
    int m = tIdx + ppIdx * BLOCK_SIZE;
    if (m < totalpairs) {
      int2 jk = pairidx_to_jk(m);
      int jj = jk.x, kk = jk.y;
      const DataT Rij = sdist[jj];
      const DataT Rik = sdist[kk];
      DataT vij_vik_dot = svec[jj].x * svec[kk].x + svec[jj].y * svec[kk].y + svec[jj].z * svec[kk].z;
      s_vij_vik_dot[tIdx] = vij_vik_dot;
      s_theta[tIdx] = acos(tc * vij_vik_dot / (Rij * Rik));
      s_vij_factor[tIdx] =
          tc / (Rij * Rij * Rij * sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rij * Rij) + Rik * Rik));
      s_vik_factor[tIdx] =
          tc / (Rik * Rik * Rik * sqrt(-tc * tc * vij_vik_dot * vij_vik_dot / (Rik * Rik) + Rij * Rij));
    }
    __syncthreads();
    // run angular calculation
    for (int nn = threadIdx.y * TILE_PER_WARP + tile.y; nn < BLOCK_SIZE; nn += BLOCK_Y * TILE_PER_WARP) {
      int n = nn + ppIdx * BLOCK_SIZE;
      float3 grad_vij = make_float3(0.f, 0.f, 0.f);
      float3 grad_vik = make_float3(0.f, 0.f, 0.f);
      int2 jk = pairidx_to_jk(n);
      int jj = jk.x, kk = jk.y;
      if (n < totalpairs) {
        const DataT Rij = sdist[jj];
        DataT fc_ij = sfc[jj];
        DataT grad_fc_ij = sfc_grad[jj];
        const DataT Rik = sdist[kk];
        DataT fc_ik = sfc[kk];
        DataT grad_fc_ik = sfc_grad[kk];

        DataT theta = s_theta[nn];
        DataT vij_vik_dot = s_vij_vik_dot[nn];
        DataT vij_factor = s_vij_factor[nn];
        DataT vik_factor = s_vik_factor[nn];

        DataT Rijk = (Rij + Rik) / 2.0f;
        DataT fc_ijk = fc_ij * fc_ik;

        IndexT subaev_offset = angular_sublength * csubaev_offsets(s_species[jj], s_species[kk], num_species);

        // iterate over ShfA and ShfZ
        for (int ishfr = tile.x; ishfr < nShfA; ishfr += TILE_SIZE) {
          DataT ShfA = __ldg(&ShfA_t[ishfr]);
          DataT factor2 = __expf(-EtaA * (Rijk - ShfA) * (Rijk - ShfA));
          DataT grad_factor2_dist = -EtaA * (Rijk - ShfA) * factor2;

          for (int itheta = 0; itheta < nShfZ; itheta++) {
            DataT ShfZ = __ldg(&ShfZ_t[itheta]);

            DataT sin_theta_ShfZ, cos_theta_ShfZ;
            __sincosf(theta - ShfZ, &sin_theta_ShfZ, &cos_theta_ShfZ);
            DataT factor1 = __powf((1 + cos_theta_ShfZ) / 2, Zeta);
            DataT grad_factor1_theta = -0.5f * Zeta * __powf((1 + cos_theta_ShfZ) / 2, Zeta - 1) * sin_theta_ShfZ;

            DataT a = grad_factor1_theta * factor2 * fc_ijk;
            DataT bj = factor1 / Rij * (grad_factor2_dist * fc_ijk + factor2 * fc_ik * grad_fc_ij);
            DataT bk = factor1 / Rik * (grad_factor2_dist * fc_ijk + factor2 * fc_ij * grad_fc_ik);

            DataT grad_vij_x =
                2.0f * (a * vij_factor * (svec[jj].x * vij_vik_dot - svec[kk].x * Rij * Rij) + svec[jj].x * bj);
            DataT grad_vij_y =
                2.0f * (a * vij_factor * (svec[jj].y * vij_vik_dot - svec[kk].y * Rij * Rij) + svec[jj].y * bj);
            DataT grad_vij_z =
                2.0f * (a * vij_factor * (svec[jj].z * vij_vik_dot - svec[kk].z * Rij * Rij) + svec[jj].z * bj);
            DataT grad_vik_x =
                2.0f * (a * vik_factor * (svec[kk].x * vij_vik_dot - svec[jj].x * Rik * Rik) + svec[kk].x * bk);
            DataT grad_vik_y =
                2.0f * (a * vik_factor * (svec[kk].y * vij_vik_dot - svec[jj].y * Rik * Rik) + svec[kk].y * bk);
            DataT grad_vik_z =
                2.0f * (a * vik_factor * (svec[kk].z * vij_vik_dot - svec[jj].z * Rik * Rik) + svec[kk].z * bk);

            if (is_double_backward) {
              int atomj_idx = atomJ[start_idx + jj];
              int atomk_idx = atomJ[start_idx + kk];
              float3 grad_force_j = grad_force_3[mol_idx * max_natoms_per_mol + atomj_idx];
              float3 grad_force_k = grad_force_3[mol_idx * max_natoms_per_mol + atomk_idx];
              grad_vij_x *= (grad_force_j.x - grad_force_i.x);
              grad_vij_y *= (grad_force_j.y - grad_force_i.y);
              grad_vij_z *= (grad_force_j.z - grad_force_i.z);
              grad_vik_x *= (grad_force_k.x - grad_force_i.x);
              grad_vik_y *= (grad_force_k.y - grad_force_i.y);
              grad_vik_z *= (grad_force_k.z - grad_force_i.z);
              atomicAdd(
                  &saev[subaev_offset + ishfr * nShfZ + itheta],
                  grad_vij_x + grad_vij_y + grad_vij_z + grad_vik_x + grad_vik_y + grad_vik_z);
            } else {
              DataT grad_output_item = saev[subaev_offset + ishfr * nShfZ + itheta];
              grad_vij_x *= grad_output_item;
              grad_vij_y *= grad_output_item;
              grad_vij_z *= grad_output_item;
              grad_vik_x *= grad_output_item;
              grad_vik_y *= grad_output_item;
              grad_vik_z *= grad_output_item;

              grad_vij.x += grad_vij_x;
              grad_vij.y += grad_vij_y;
              grad_vij.z += grad_vij_z;
              grad_vik.x += grad_vik_x;
              grad_vik.y += grad_vik_y;
              grad_vik.z += grad_vik_z;
            }
          }
        }
      }
      // accumulate gradients to share memory for backward
      if (!is_double_backward) {
        spos_i_grad.x += (-grad_vij.x - grad_vik.x);
        spos_i_grad.y += (-grad_vij.y - grad_vik.y);
        spos_i_grad.z += (-grad_vij.z - grad_vik.z);

        for (int offset = TILE_SIZE / 2; offset > 0; offset /= 2) {
          grad_vij.x += __shfl_down_sync(0xFFFFFFFF, grad_vij.x, offset);
          grad_vij.y += __shfl_down_sync(0xFFFFFFFF, grad_vij.y, offset);
          grad_vij.z += __shfl_down_sync(0xFFFFFFFF, grad_vij.z, offset);
          grad_vik.x += __shfl_down_sync(0xFFFFFFFF, grad_vik.x, offset);
          grad_vik.y += __shfl_down_sync(0xFFFFFFFF, grad_vik.y, offset);
          grad_vik.z += __shfl_down_sync(0xFFFFFFFF, grad_vik.z, offset);
        }

        // TODO this is a bottleneck
        // 0.6ms/2.9ms = 20.1% (2.9ms is the total backward timing)
        // bank confilct or atomicAdd?
        if (n < totalpairs && laneIdx % TILE_SIZE == 0) {
          atomicAdd(&spos_j_grad[jj].x, grad_vij.x);
          atomicAdd(&spos_j_grad[jj].y, grad_vij.y);
          atomicAdd(&spos_j_grad[jj].z, grad_vij.z);

          atomicAdd(&spos_j_grad[kk].x, grad_vik.x);
          atomicAdd(&spos_j_grad[kk].y, grad_vik.y);
          atomicAdd(&spos_j_grad[kk].z, grad_vik.z);
        }
      }
    }
    __syncthreads();
  }

  if (is_double_backward) {
    auto& grad_grad_aev = grad_input;
    for (int iaev = tIdx; iaev < angular_length; iaev += blockDim.x * blockDim.y) {
      grad_grad_aev[mol_idx][i][radial_length + iaev] = saev[iaev];
    }
  } else {
    auto& grad_coord = grad_input;
    int atomi_idx = i;
    for (int offset = 16; offset > 0; offset /= 2) {
      spos_i_grad.x += __shfl_down_sync(0xFFFFFFFF, spos_i_grad.x, offset);
      spos_i_grad.y += __shfl_down_sync(0xFFFFFFFF, spos_i_grad.y, offset);
      spos_i_grad.z += __shfl_down_sync(0xFFFFFFFF, spos_i_grad.z, offset);
    }
    if (laneIdx == 0) {
      atomicAdd(&grad_coord[mol_idx][atomi_idx][0], spos_i_grad.x);
      atomicAdd(&grad_coord[mol_idx][atomi_idx][1], spos_i_grad.y);
      atomicAdd(&grad_coord[mol_idx][atomi_idx][2], spos_i_grad.z);
    }

    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      int atomj_idx = atomJ[start_idx + jj];
      atomicAdd(&grad_coord[mol_idx][atomj_idx][0], spos_j_grad[jj].x);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][1], spos_j_grad[jj].y);
      atomicAdd(&grad_coord[mol_idx][atomj_idx][2], spos_j_grad[jj].z);
    }
  }
}

template <bool use_cos_cutoff, typename SpeciesT, typename DataT>
__global__ void cuRadialAEVs(
    torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits> aev_t,
    const int* __restrict__ atomJ,
    const float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ numJPerI,
    const int* __restrict__ startIdxJ,
    float Rcr,
    int radial_length,
    int radial_sublength,
    int nRadialRij,
    int max_numPairsPerAtom) {
  extern __shared__ DataT smem[];

  DataT* s_radial = &smem[0];
  DataT* s_fc = &smem[radial_length];

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  int start_idx = startIdxJ[cIdx];
  int jnum = numJPerI[cIdx];

  if (jnum < 1)
    return;

  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
    s_radial[iaev] = 0;
  }

  for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    if (use_cos_cutoff)
      s_fc[jj] = cosine_cutoff_fwd(Rij, Rcr);
    else
      s_fc[jj] = smooth_cutoff_fwd(Rij, Rcr);
  }
  __syncthreads();

  for (int jj = threadIdx.y; jj < jnum; jj += blockDim.y) {
    DataT fc = s_fc[jj];
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    SpeciesT specie_j = species_t[mol_idx][j];

    for (int ishfr = threadIdx.x; ishfr < nShfR; ishfr += blockDim.x) {
      DataT ShfR = __ldg(&ShfR_t[ishfr]);
      DataT GmR = 0.25f * __expf(-EtaR * (Rij - ShfR) * (Rij - ShfR)) * fc;
      atomicAdd(&s_radial[specie_j * radial_sublength + ishfr], GmR);
    }
  }

  __syncthreads();
  for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
    aev_t[mol_idx][i][iaev] = s_radial[iaev];
  }
}

// every <TILE_SIZE> threads take care of 1 RIJ, and iterate for <nShfR / TILE_SIZE> times
template <bool is_double_backward, int TILE_SIZE, bool use_cos_cutoff, typename SpeciesT, typename DataT>
__global__ void cuRadialAEVs_backward_or_doublebackward(
    const DataT* pos_p,
    const torch::PackedTensorAccessor32<SpeciesT, 2, torch::RestrictPtrTraits> species_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> ShfR_t,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> EtaR_t,
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_aev, // daev for backward, ddaev for double backward
    torch::PackedTensorAccessor32<DataT, 3, torch::RestrictPtrTraits>
        grad_coord_or_force, // dcoord for backward, dforce(i.e. ddcoord) for double backward
    const torch::PackedTensorAccessor32<SpeciesT, 1, torch::RestrictPtrTraits> atomJ,
    const torch::PackedTensorAccessor32<DataT, 1, torch::RestrictPtrTraits> distJ,
    const AtomI* __restrict__ atom_i,
    const int* __restrict__ numJPerI,
    const int* __restrict__ startIdxJ,
    float Rcr,
    int radial_length,
    int radial_sublength,
    int nRadialRij,
    int max_numPairsPerAtom) {
  extern __shared__ DataT smem[];
  DataT* s_grad_dist = &smem[0]; // ddist for backward, dddist for double backward
  DataT* saev = &smem[max_numPairsPerAtom];
  const float3* pos_p_3 = reinterpret_cast<const float3*>(pos_p);
  const int max_natoms_per_mol = species_t.size(1);

  int cIdx = blockIdx.x; // central atom id
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x; // local thread idx

  int nShfR = ShfR_t.size(0);
  DataT EtaR = EtaR_t[0];

  int start_idx = startIdxJ[cIdx];
  int jnum = numJPerI[cIdx];

  if (jnum < 1)
    return;

  AtomI aI = atom_i[cIdx];
  int mol_idx = aI.midx;
  int i = aI.i;

  float3 pos_i = pos_p_3[mol_idx * max_natoms_per_mol + i];

  if (is_double_backward) {
    float3* grad_force = reinterpret_cast<float3*>(&grad_coord_or_force[0][0][0]);
    float3 dforce_i = grad_force[mol_idx * max_natoms_per_mol + i];
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      DataT Rij = distJ[start_idx + jj];
      int j = atomJ[start_idx + jj];
      float3 pos_j = pos_p_3[mol_idx * max_natoms_per_mol + j];
      float3 dforce_j = grad_force[mol_idx * max_natoms_per_mol + j];
      float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);
      s_grad_dist[jj] = ((dforce_j.x - dforce_i.x) * delta.x + (dforce_j.y - dforce_i.y) * delta.y +
                         (dforce_j.z - dforce_i.z) * delta.z) /
          Rij;
    }
    for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
      saev[iaev] = 0;
    }
  } else {
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      s_grad_dist[jj] = 0;
    }
    for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
      saev[iaev] = grad_aev[mol_idx][i][iaev];
    }
  }

  __syncthreads();

  DataT upstream_grad;
  for (int jj = threadIdx.y; jj < jnum; jj += blockDim.y) {
    DataT Rij = distJ[start_idx + jj];
    int j = atomJ[start_idx + jj];
    DataT fc;
    DataT fc_grad;
    if (use_cos_cutoff) {
      fc = cosine_cutoff_fwd(Rij, Rcr);
      fc_grad = cosine_cutoff_bwd(Rij, Rcr);
    } else {
      fc = smooth_cutoff_fwd(Rij, Rcr);
      fc_grad = smooth_cutoff_bwd(Rij, Rcr);
    }
    SpeciesT specie_j = species_t[mol_idx][j];

    if (is_double_backward) {
      upstream_grad = s_grad_dist[jj];
    }

    for (int ishfr = threadIdx.x; ishfr < nShfR; ishfr += blockDim.x) {
      DataT ShfR = __ldg(&ShfR_t[ishfr]);

      DataT GmR = 0.25f * __expf(-EtaR * (Rij - ShfR) * (Rij - ShfR));
      DataT GmR_grad = -EtaR * (-2.0f * ShfR + 2.0f * Rij) * GmR;
      DataT jacobian = GmR_grad * fc + GmR * fc_grad;

      if (is_double_backward) {
        atomicAdd(&saev[specie_j * radial_sublength + ishfr], upstream_grad * jacobian);
      } else {
        upstream_grad = saev[specie_j * radial_sublength + ishfr];
        atomicAdd(&s_grad_dist[jj], upstream_grad * jacobian);
      }
    }
  }

  __syncthreads();
  if (is_double_backward) {
    auto& grad_grad_aev = grad_aev;
    for (int iaev = tIdx; iaev < radial_length; iaev += blockDim.x * blockDim.y) {
      grad_grad_aev[mol_idx][i][iaev] = saev[iaev];
    }
  } else {
    auto& grad_coord = grad_coord_or_force;
    for (int jj = tIdx; jj < jnum; jj += blockDim.x * blockDim.y) {
      DataT Rij = distJ[start_idx + jj];
      int j = atomJ[start_idx + jj];
      float3 pos_j = pos_p_3[mol_idx * max_natoms_per_mol + j];
      float3 delta = make_float3(pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z);

      DataT grad_dist_coord_x = delta.x / Rij;
      DataT grad_dist_coord_y = delta.y / Rij;
      DataT grad_dist_coord_z = delta.z / Rij;
      DataT grad_radial_dist_item = s_grad_dist[jj];

      atomicAdd(&grad_coord[mol_idx][j][0], grad_radial_dist_item * grad_dist_coord_x);
      atomicAdd(&grad_coord[mol_idx][j][1], grad_radial_dist_item * grad_dist_coord_y);
      atomicAdd(&grad_coord[mol_idx][j][2], grad_radial_dist_item * grad_dist_coord_z);

      atomicAdd(&grad_coord[mol_idx][i][0], -grad_radial_dist_item * grad_dist_coord_x);
      atomicAdd(&grad_coord[mol_idx][i][1], -grad_radial_dist_item * grad_dist_coord_y);
      atomicAdd(&grad_coord[mol_idx][i][2], -grad_radial_dist_item * grad_dist_coord_z);
    }
  }
}

// There are 2 tasks in the post process:
// 1. Copy radial's neighbor from tmp buffer
// 2. And check whether it is within Rca, if so then also copy it into angular's neighbor list
template <int ATOM_I_PER_BLOCK>
__global__ void postProcessNbrList(
    const int* __restrict__ atomJ,
    const float* __restrict__ distJ,
    const AtomI* __restrict__ atom_i,
    int* __restrict__ radial_atomJ,
    float* __restrict__ radial_distJ,
    int* __restrict__ angular_atomJ,
    float* __restrict__ angular_distJ,
    const int* __restrict__ radial_numJPerI,
    const int* __restrict__ startIdxPerI,
    float Rca,
    int* __restrict__ angular_numJPerI,
    int num_atomI,
    int max_natoms_per_mol,
    int max_numj_per_i_in_Rcr) {
  __shared__ int s_new_pcounter_i[ATOM_I_PER_BLOCK];
  __shared__ int s_num_max;
  int gi = blockIdx.x * blockDim.y + threadIdx.y;
  if (gi >= num_atomI)
    return;

  AtomI aI = atom_i[gi];
  int i = aI.i;
  int mol_idx = aI.midx;
  int jnum = radial_numJPerI[gi];
  int start_i = startIdxPerI[gi];
  int ii = threadIdx.y;
  int idx = blockDim.x * threadIdx.y + threadIdx.x;
  int pairs_per_mol = max_natoms_per_mol * max_numj_per_i_in_Rcr;

  if (idx < ATOM_I_PER_BLOCK) {
    s_new_pcounter_i[idx] = 0;
    int ii = blockIdx.x * blockDim.y + idx;
    int num_max = ii < num_atomI ? radial_numJPerI[ii] : 0;

    for (int offset = 16; offset > 0; offset /= 2) {
      num_max = max(num_max, __shfl_down_sync(0xFFFFFFFF, num_max, offset));
    }
    if (idx == 0) {
      s_num_max = num_max;
    }
  }
  __syncthreads();

  for (int jj = threadIdx.x; jj < s_num_max && jj < jnum; jj += blockDim.x) {
    int j = atomJ[pairs_per_mol * mol_idx + i * max_numj_per_i_in_Rcr + jj];
    float dist = distJ[pairs_per_mol * mol_idx + i * max_numj_per_i_in_Rcr + jj];
    radial_atomJ[start_i + jj] = j;
    radial_distJ[start_i + jj] = dist;
    if (dist <= Rca) {
      int pidx = atomicAdd(&s_new_pcounter_i[ii], 1);
      angular_atomJ[start_i + pidx] = j;
      angular_distJ[start_i + pidx] = dist;
    }
  }
  __syncthreads();

  gi = idx + blockIdx.x * blockDim.y;
  if (idx < blockDim.y && gi < num_atomI) {
    angular_numJPerI[gi] = s_new_pcounter_i[idx];
  }
}

// NOTE: assumes size of EtaA_t = Zeta_t = EtaR_t = 1
template <bool use_cos_cutoff>
void cuaev_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const AEVScalarParams& aev_params,
    Result& result) {
  TORCH_CHECK(
      (species_t.dtype() == torch::kInt32) && (coordinates_t.dtype() == torch::kFloat32), "Unsupported input type");
  TORCH_CHECK(
      aev_params.EtaR_t.size(0) == 1 && aev_params.EtaA_t.size(0) == 1 && aev_params.Zeta_t.size(0) == 1,
      "cuda extension is currently not supported for the specified "
      "configuration");
  TORCH_CHECK(
      coordinates_t.device() == species_t.device() && coordinates_t.device() == aev_params.EtaR_t.device() &&
          coordinates_t.device() == aev_params.EtaA_t.device(),
      "coordinates, species, and aev_params should be on the same device");

  float Rcr = aev_params.Rcr;
  float Rca = aev_params.Rca;
  const int n_molecules = species_t.size(0);
  const int max_natoms_per_mol = species_t.size(1);
  int aev_length = aev_params.radial_length + aev_params.angular_length;
  int total_atoms = n_molecules * max_natoms_per_mol;
  float* coordinates_p = (float*)coordinates_t.data_ptr();
  TORCH_CHECK(coordinates_t.is_contiguous(), "Coordinate data is not contiguous");

  // TODO replace zeros with empty
  result.aev_t = torch::zeros({n_molecules, max_natoms_per_mol, aev_length}, coordinates_t.options());
  if (species_t.numel() == 0) {
    return;
  }

  // set cuda device and stream
  at::cuda::CUDAGuard device_guard(coordinates_t.device().index());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  at::globalContext().lazyInitCUDA();

  int max_numj_per_i_in_Rcr = min(max_natoms_per_mol, MAX_NUMJ_PER_I_IN_RCR);
  int pairs_per_mol = max_natoms_per_mol * max_numj_per_i_in_Rcr;
  auto total_natom_pairs = n_molecules * pairs_per_mol;
  auto d_options = torch::dtype(torch::kUInt8).device(coordinates_t.device());

  // buffer to store all the pairwise distance (Rij)
  Tensor atomJ_t = torch::empty(total_natom_pairs, d_options.dtype(torch::kInt32));
  int* atomJ_p = (int*)atomJ_t.data_ptr();
  Tensor distJ_t = torch::empty(total_natom_pairs, d_options.dtype(torch::kFloat32));
  float* distJ_p = (float*)distJ_t.data_ptr();

  // radial and angular share the same data of atomI, startIdxJ and nI
  result.atomI_t = torch::empty(total_atoms * 2, d_options.dtype(torch::kInt32));
  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  result.startIdxJ_t = torch::empty(total_atoms, d_options.dtype(torch::kInt32));
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();

  // radial and angular numJPerI counter
  // radial_num_per_atom ranges from 10 - 60
  result.radialNbr.numJPerI_t = torch::zeros(total_atoms, d_options.dtype(torch::kInt32));
  int* radialNbr_numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();
  result.angularNbr.numJPerI_t = torch::zeros(total_atoms, d_options.dtype(torch::kInt32));
  int* angularNbr_numJPerI_p = (int*)result.angularNbr.numJPerI_t.data_ptr();

  constexpr int ATOM_I_PER_BLOCK = 32;
  // single molecule mode
  if (n_molecules == 1) {
#ifdef TORCHANI_DEBUG
    setlocale(LC_ALL, "");
    printf("\n%-35s %'d atoms\n", "single molecule", max_natoms_per_mol);
#endif
    constexpr int ATOM_J_PER_TILE = 32;
    int blocks = (total_atoms + ATOM_I_PER_BLOCK - 1) / ATOM_I_PER_BLOCK;
    dim3 block(ATOM_J_PER_TILE, ATOM_I_PER_BLOCK, 1);
    pairwiseDistanceSingleMolecule<ATOM_I_PER_BLOCK, ATOM_J_PER_TILE><<<blocks, block, 0, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_p,
        result.radialNbr.numJPerI_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        atomI_p,
        atomJ_p,
        distJ_p,
        Rcr,
        max_natoms_per_mol,
        max_numj_per_i_in_Rcr);
    result.nI = total_atoms;
  } else { // batch mode
    // tmp storage because of padding
#ifdef TORCHANI_DEBUG
    setlocale(LC_ALL, "");
    printf("\n%-35s %d molecules, total %'d atoms\n", "batch molecules", n_molecules, total_atoms);
#endif
    Tensor numJPerI_t = torch::empty(total_atoms, d_options.dtype(torch::kInt32));
    int* numJPerI_p = (int*)numJPerI_t.data_ptr();
    Tensor atom_i_t = torch::empty(total_atoms * 2, d_options.dtype(torch::kInt32));
    AtomI* atom_i_p = (AtomI*)atom_i_t.data_ptr();

    dim3 block(8, 16, 1);
    // maximum 4096 atoms, which needs 49152 byte (48 kb) of shared memory
    int smem_pairdist = sizeof(float) * max_natoms_per_mol * 5; // x, y, z, spe, counter
    pairwiseDistance<<<n_molecules, block, smem_pairdist, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_p,
        numJPerI_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        atom_i_p,
        atomJ_p,
        distJ_p,
        Rcr,
        max_natoms_per_mol,
        max_numj_per_i_in_Rcr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // remove padding numJPerI if numj == 0
    result.nI = cubDeviceSelectIf(
        numJPerI_p, radialNbr_numJPerI_p, total_atoms, [=] __device__(const int numj) { return (bool)numj; }, stream);

    // also remove padding atomI
    // Note: cub::DeviceSelect::Flagged Bug: flag current only allow bool or int which is ether 0 or 1
    // https://github.com/NVIDIA/cub/issues/235
    auto flags_t = numJPerI_t.to(torch::kBool);
    char* flags_p = (char*)flags_t.data_ptr();
    cubDeviceSelectFlagged(atom_i_p, atomI_p, total_atoms, flags_p, stream);
  }

  cubScan(radialNbr_numJPerI_p, startIdxJ_p, total_atoms, stream);
  result.radialNbr.nJ = (result.startIdxJ_t[-1] + result.radialNbr.numJPerI_t[-1]).item<int>();

#ifdef TORCHANI_DEBUG
  printf("%-35s %'d\n", "nI", result.nI);
  printf("%-35s %'d\n", "radialNbr  nJ", result.radialNbr.nJ);
#endif

  result.radialNbr.atomJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kInt32));
  int* radialNbr_atomJ_p = (int*)result.radialNbr.atomJ_t.data_ptr();
  result.radialNbr.distJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kFloat32));
  float* radialNbr_distJ_p = (float*)result.radialNbr.distJ_t.data_ptr();

  result.angularNbr.atomJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kInt32));
  int* angularNbr_atomJ_p = (int*)result.angularNbr.atomJ_t.data_ptr();
  result.angularNbr.distJ_t = torch::empty(result.radialNbr.nJ, d_options.dtype(torch::kFloat32));
  float* angularNbr_distJ_p = (float*)result.angularNbr.distJ_t.data_ptr();

  { // postProcessNbrList
    int ATOM_J_PER_TILE = 16;
    dim3 block(ATOM_J_PER_TILE, ATOM_I_PER_BLOCK, 1);
    int blocks = (result.nI + ATOM_I_PER_BLOCK - 1) / ATOM_I_PER_BLOCK;
    postProcessNbrList<ATOM_I_PER_BLOCK><<<blocks, block, 0, stream>>>(
        atomJ_p,
        distJ_p,
        atomI_p,
        radialNbr_atomJ_p,
        radialNbr_distJ_p,
        angularNbr_atomJ_p,
        angularNbr_distJ_p,
        radialNbr_numJPerI_p,
        startIdxJ_p,
        Rca,
        angularNbr_numJPerI_p,
        result.nI,
        max_natoms_per_mol,
        max_numj_per_i_in_Rcr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef TORCHANI_DEBUG
    result.angularNbr.nJ = cubSum(angularNbr_numJPerI_p, result.nI, stream);
    printf("%-35s %'d\n", "angularNbr nJ", result.angularNbr.nJ);
#endif
  }

  // Merge two cubMax streamSync into one
  result.radialNbr.maxNumJPerI = cubMax(radialNbr_numJPerI_p, result.nI, stream, /* sync */ false);
  result.angularNbr.maxNumJPerI = cubMax(angularNbr_numJPerI_p, result.nI, stream, /* sync */ false);
  cudaStreamSynchronize(stream);

  { // RadialAEV
    constexpr dim3 block_radial(8, 16, 1);
    int smem_radial = aev_params.radial_length * sizeof(float) + result.radialNbr.maxNumJPerI * sizeof(float);
    cuRadialAEVs<use_cos_cutoff><<<result.nI, block_radial, smem_radial, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        result.aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        radialNbr_atomJ_p,
        radialNbr_distJ_p,
        atomI_p,
        radialNbr_numJPerI_p,
        startIdxJ_p,
        aev_params.Rcr,
        aev_params.radial_length,
        aev_params.radial_sublength,
        result.radialNbr.nJ,
        result.radialNbr.maxNumJPerI);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef TORCHANI_DEBUG
    printf("%-35s %d\n", "radialNbr  maxNumJPerI", result.radialNbr.maxNumJPerI);
#endif
  }

  { // AngularAEV
    auto cal_smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
      int sm_aev = sizeof(float) * aev_params.angular_length;
      int sxyz = sizeof(float) * max_nbrs * 3;
      int sRij = sizeof(float) * max_nbrs;
      int sfc = sizeof(float) * max_nbrs;
      int sj = sizeof(int) * max_nbrs;
      return (sm_aev + sxyz + sRij + sfc + sj) * ncatom_per_tpb;
    };

    int smem_size = cal_smem_size(result.angularNbr.maxNumJPerI, 1);
    // temporary fix because of nvcc constexpr BUG
    constexpr int block_x = C10_WARP_SIZE;
    constexpr int block_y = 4;
    constexpr dim3 block(block_x, block_y, 1);
    cuAngularAEVs<block_x, block_y, use_cos_cutoff><<<result.nI, block, smem_size, stream>>>(
        species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        coordinates_p,
        aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        result.aev_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        angularNbr_atomJ_p,
        angularNbr_distJ_p,
        atomI_p,
        angularNbr_numJPerI_p,
        startIdxJ_p,
        aev_params.Rca,
        aev_params.angular_length,
        aev_params.angular_sublength,
        aev_params.radial_length,
        aev_params.num_species,
        result.angularNbr.maxNumJPerI,
        result.nI);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef TORCHANI_DEBUG
    printf("%-35s %d\n", "angularNbr maxNumJPerI", result.angularNbr.maxNumJPerI);
    printf("%-35s %'d bytes\n", "forward  angular smem_size", smem_size);
#endif
  }
}

template <bool use_cos_cutoff>
Tensor cuaev_backward(const Tensor& grad_output, const AEVScalarParams& aev_params, const Result& result) {
  using namespace torch::indexing;
  Tensor coordinates_t = result.coordinates_t;
  Tensor species_t = result.species_t;

  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  at::cuda::CUDAGuard device_guard(coordinates_t.device().index());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  auto grad_coord = torch::zeros(coordinates_t.sizes(), coordinates_t.options().requires_grad(false)); // [2, 5, 3]

  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  int* angular_numJPerI_p = (int*)result.angularNbr.numJPerI_t.data_ptr();
  int* radial_numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();
  float* coordinates_p = (float*)coordinates_t.data_ptr();

  // radial
  constexpr dim3 block_radial(8, 16, 1);
  int smem_radial =
      result.radialNbr.maxNumJPerI * sizeof(float) + aev_params.radial_length * sizeof(float); // grad_dist, grad_aev
  cuRadialAEVs_backward_or_doublebackward<false, 8, use_cos_cutoff><<<result.nI, block_radial, smem_radial, stream>>>(
      coordinates_p,
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.radialNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.radialNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      radial_numJPerI_p,
      startIdxJ_p,
      aev_params.Rcr,
      aev_params.radial_length,
      aev_params.radial_sublength,
      result.radialNbr.nJ,
      result.radialNbr.maxNumJPerI);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // angular
  auto cal_smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sm_aev = sizeof(float) * (aev_params.angular_length);
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;
    return sm_aev + (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };
  int smem_size = cal_smem_size(result.angularNbr.maxNumJPerI, 1);

#ifdef TORCHANI_DEBUG
  printf("%-35s %'d bytes\n", "backward angular smem_size", smem_size);
#endif

  constexpr int block_x = C10_WARP_SIZE;
  constexpr int block_y = 4;
  constexpr dim3 block(block_x, block_y, 1);
  cuAngularAEVs_backward_or_doublebackward<false, block_x, block_y, use_cos_cutoff>
      <<<result.nI, block, smem_size, stream>>>(
          species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
          coordinates_p,
          aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          grad_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          grad_coord.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          result.angularNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
          result.angularNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          atomI_p,
          angular_numJPerI_p,
          startIdxJ_p,
          aev_params.Rca,
          aev_params.angular_length,
          aev_params.angular_sublength,
          aev_params.radial_length,
          aev_params.num_species,
          result.angularNbr.maxNumJPerI,
          result.nI);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_coord;
}

template <bool use_cos_cutoff>
Tensor cuaev_double_backward(const Tensor& grad_force, const AEVScalarParams& aev_params, const Result& result) {
  using namespace torch::indexing;
  Tensor coordinates_t = result.coordinates_t;
  Tensor species_t = result.species_t;

  const int n_molecules = coordinates_t.size(0);
  const int max_natoms_per_mol = coordinates_t.size(1);
  at::cuda::CUDAGuard device_guard(coordinates_t.device().index());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  int aev_length = aev_params.radial_length + aev_params.angular_length;

  auto grad_grad_aev = torch::zeros(
      {coordinates_t.size(0), coordinates_t.size(1), aev_length},
      coordinates_t.options().requires_grad(false)); // [2, 5, 384]

  AtomI* atomI_p = (AtomI*)result.atomI_t.data_ptr();
  int* angular_numJPerI_p = (int*)result.angularNbr.numJPerI_t.data_ptr();
  int* radial_numJPerI_p = (int*)result.radialNbr.numJPerI_t.data_ptr();
  int* startIdxJ_p = (int*)result.startIdxJ_t.data_ptr();
  float* coordinates_p = (float*)coordinates_t.data_ptr();
  TORCH_CHECK(grad_force.is_contiguous(), "grad_force's data is not contiguous");

  // radial
  constexpr dim3 block_radial(8, 16, 1);
  int smem_radial = result.radialNbr.maxNumJPerI * sizeof(float) +
      aev_params.radial_length * sizeof(float); // grad_dist, grad_grad_aev
  cuRadialAEVs_backward_or_doublebackward<true, 8, use_cos_cutoff><<<result.nI, block_radial, smem_radial, stream>>>(
      coordinates_p,
      species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
      aev_params.ShfR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      aev_params.EtaR_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      result.radialNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
      result.radialNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      atomI_p,
      radial_numJPerI_p,
      startIdxJ_p,
      aev_params.Rcr,
      aev_params.radial_length,
      aev_params.radial_sublength,
      result.radialNbr.nJ,
      result.radialNbr.maxNumJPerI);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // angular
  auto cal_smem_size = [&aev_params](int max_nbrs, int ncatom_per_tpb) {
    int sm_aev = sizeof(float) * aev_params.angular_length;
    int sxyz = sizeof(float) * max_nbrs * 3;
    int sj_xyz_grad = sizeof(float) * max_nbrs * 3;
    int sRij = sizeof(float) * max_nbrs;
    int sfc = sizeof(float) * max_nbrs;
    int sfc_grad = sizeof(float) * max_nbrs;
    int sj = sizeof(int) * max_nbrs;
    return sm_aev + (sxyz + sj_xyz_grad + sRij + sfc + sfc_grad + sj) * ncatom_per_tpb;
  };
  int smem_size = cal_smem_size(result.angularNbr.maxNumJPerI, 1);
  constexpr int block_x = C10_WARP_SIZE;
  constexpr int block_y = 4;
  constexpr dim3 block(block_x, block_y, 1);
  cuAngularAEVs_backward_or_doublebackward<true, block_x, block_y, use_cos_cutoff>
      <<<result.nI, block, smem_size, stream>>>(
          species_t.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
          coordinates_p,
          aev_params.ShfA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.ShfZ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.EtaA_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          aev_params.Zeta_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          grad_force.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          grad_grad_aev.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          result.angularNbr.atomJ_t.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
          result.angularNbr.distJ_t.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          atomI_p,
          angular_numJPerI_p,
          startIdxJ_p,
          aev_params.Rca,
          aev_params.angular_length,
          aev_params.angular_sublength,
          aev_params.radial_length,
          aev_params.num_species,
          result.angularNbr.maxNumJPerI,
          result.nI);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return grad_grad_aev;
}

// Explicit Template Instantiation
Result CuaevComputer::forward(const Tensor& coordinates_t, const Tensor& species_t) {
  Result result(coordinates_t, species_t);
  if (aev_params.use_cos_cutoff)
    cuaev_forward<true>(coordinates_t, species_t, aev_params, result);
  else
    cuaev_forward<false>(coordinates_t, species_t, aev_params, result);
  return result;
}

Tensor CuaevComputer::backward(const Tensor& grad_e_aev, const Result& result) {
  if (aev_params.use_cos_cutoff)
    return cuaev_backward<true>(grad_e_aev, aev_params, result); // force
  else
    return cuaev_backward<false>(grad_e_aev, aev_params, result); // force
}

Tensor CuaevComputer::double_backward(const Tensor& grad_force, const Result& result) {
  if (aev_params.use_cos_cutoff)
    return cuaev_double_backward<true>(grad_force, aev_params, result); // grad_grad_aev
  else
    return cuaev_double_backward<false>(grad_force, aev_params, result); // grad_grad_aev
}
