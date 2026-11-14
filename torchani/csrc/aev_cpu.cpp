#include <c10/util/MathConstants.h>
#include <torch/extension.h>

#include "aev_cpu.h"

using torch::Tensor;
using torch::indexing::Slice;

static Tensor cutoff_fn(const Tensor& distances, double cutoff, bool use_cos_cutoff) {
  if (use_cos_cutoff) {
    return 0.5 * torch::cos(distances * (c10::pi<double> / cutoff)) + 0.5;
  }
  Tensor exponent = 1 - 1 / (1 - (distances / cutoff).pow(2)).clamp_min(1e-10);
  return torch::exp(exponent);
}

// Input: directed pairs (center, side), with deltas from center to side
// Output shape: (total-atoms, num-species * radial-feats)
static Tensor radial_aev(
    const Tensor& species_flat,
    const Tensor& centers,
    const Tensor& sides,
    const Tensor& distances,
    double Rcr,
    const Tensor& EtaR,
    const Tensor& ShfR,
    int64_t num_species,
    bool use_cos_cutoff) {
  int64_t total_atoms = species_flat.size(0);
  int64_t num_feats = ShfR.size(0);
  Tensor terms = 0.25 * torch::exp(-EtaR * (distances.view({-1, 1}) - ShfR.view({1, -1})).pow(2));
  terms = terms * cutoff_fn(distances, Rcr, use_cos_cutoff).view({-1, 1});

  Tensor aev = torch::zeros({total_atoms * num_species, num_feats}, terms.options());
  Tensor index = centers * num_species + species_flat.index_select(0, sides);
  aev.index_add_(0, index, terms);
  return aev.view({total_atoms, num_species * num_feats});
}

// Input: directed pairs (center, side), with deltas from center to side
// Output shape: (total-atoms, num-species-pairs * angular-feats)
static Tensor angular_aev(
    const Tensor& species_flat,
    const Tensor& centers,
    const Tensor& sides,
    const Tensor& deltas,
    const Tensor& distances,
    double Rca,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  int64_t total_atoms = species_flat.size(0);
  int64_t num_species_pairs = num_species * (num_species + 1) / 2;
  int64_t num_feats = ShfA.size(0) * ShfZ.size(0);
  auto long_opts = centers.options();

  Tensor in_cutoff = (distances.detach() <= Rca).nonzero().view(-1);
  Tensor order = std::get<1>(centers.index_select(0, in_cutoff).sort(/*stable=*/true, /*dim=*/0, /*descending=*/false));
  Tensor pair_idxs = in_cutoff.index_select(0, order);
  Tensor sorted_centers = centers.index_select(0, pair_idxs);

  // For each center, all pairs of its (local) side atoms
  Tensor counts = std::get<2>(torch::unique_consecutive(sorted_centers, false, true));
  int64_t max_count = counts.numel() == 0 ? 0 : counts.max().item<int64_t>();
  Tensor pair_sizes = torch::div(counts * (counts - 1), 2, "floor");
  Tensor local_pairs = torch::tril_indices(max_count, max_count, -1, long_opts);
  Tensor mask = torch::arange(local_pairs.size(1), long_opts).view({1, -1}) < pair_sizes.view({-1, 1});
  Tensor starts = torch::cumsum(counts, 0) - counts;
  Tensor triple_idxs = local_pairs.view({2, 1, -1}) + starts.view({1, -1, 1});
  triple_idxs = triple_idxs.index({Slice(), mask});
  Tensor idxs_j = pair_idxs.index_select(0, triple_idxs[0]);
  Tensor idxs_k = pair_idxs.index_select(0, triple_idxs[1]);

  Tensor central = centers.index_select(0, idxs_j);
  Tensor species_j = species_flat.index_select(0, sides.index_select(0, idxs_j));
  Tensor species_k = species_flat.index_select(0, sides.index_select(0, idxs_k));
  Tensor deltas_j = deltas.index_select(0, idxs_j);
  Tensor deltas_k = deltas.index_select(0, idxs_k);
  Tensor distances_j = distances.index_select(0, idxs_j);
  Tensor distances_k = distances.index_select(0, idxs_k);

  Tensor cos_angles = (deltas_j * deltas_k).sum(-1) / torch::clamp_min(distances_j * distances_k, 1e-10);
  Tensor angles = torch::acos(0.95 * cos_angles);
  Tensor angular_terms = 2 * ((1 + torch::cos(angles.view({-1, 1}) - ShfZ.view({1, -1}))) / 2).pow(Zeta);
  Tensor mean_distances = (distances_j + distances_k) / 2;
  Tensor radial_terms = torch::exp(-EtaA * (mean_distances.view({-1, 1}) - ShfA.view({1, -1})).pow(2));
  Tensor terms = (radial_terms.unsqueeze(2) * angular_terms.unsqueeze(1)).view({-1, num_feats});
  Tensor factor = cutoff_fn(distances_j, Rca, use_cos_cutoff) * cutoff_fn(distances_k, Rca, use_cos_cutoff);
  terms = terms * factor.view({-1, 1});

  // Index of the species pair in the upper triangle (with diagonal) of a (S, S) matrix
  Tensor lower = torch::minimum(species_j, species_k);
  Tensor upper = torch::maximum(species_j, species_k);
  Tensor species_pair = lower * num_species - torch::div(lower * (lower - 1), 2, "floor") + upper - lower;

  Tensor aev = torch::zeros({total_atoms * num_species_pairs, num_feats}, terms.options());
  aev.index_add_(0, central * num_species_pairs + species_pair, terms);
  return aev.view({total_atoms, num_species_pairs * num_feats});
}

// Input: directed pairs (center, side), with deltas from center to side
// Output shape: (molecules, atoms, aev-length)
static Tensor compute_aev(
    const Tensor& species,
    const Tensor& centers,
    const Tensor& sides,
    const Tensor& deltas,
    double Rcr,
    double Rca,
    const Tensor& EtaR,
    const Tensor& ShfR,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  auto dtype = deltas.scalar_type();
  int64_t num_species_pairs = num_species * (num_species + 1) / 2;
  int64_t aev_length = num_species * ShfR.size(0) + num_species_pairs * ShfA.size(0) * ShfZ.size(0);
  Tensor species_flat = species.reshape(-1).to(torch::kLong);

  Tensor in_cutoff = (deltas.detach().norm(2, -1) <= Rcr).nonzero().view(-1);
  Tensor cut_centers = centers.index_select(0, in_cutoff);
  Tensor cut_sides = sides.index_select(0, in_cutoff);
  Tensor cut_deltas = deltas.index_select(0, in_cutoff);
  Tensor distances = cut_deltas.norm(2, -1);

  Tensor radial = radial_aev(
      species_flat,
      cut_centers,
      cut_sides,
      distances,
      Rcr,
      EtaR.to(dtype),
      ShfR.to(dtype),
      num_species,
      use_cos_cutoff);
  Tensor angular = angular_aev(
      species_flat,
      cut_centers,
      cut_sides,
      cut_deltas,
      distances,
      Rca,
      EtaA.to(dtype),
      Zeta.to(dtype),
      ShfA.to(dtype),
      ShfZ.to(dtype),
      num_species,
      use_cos_cutoff);
  return torch::cat({radial, angular}, -1).view({species.size(0), species.size(1), aev_length});
}

// Input: half pairs (2, P), with deltas_01 from atom pairs[0] to atom pairs[1]
static Tensor compute_aev_from_half_pairs(
    const Tensor& species,
    const Tensor& pairs,
    const Tensor& deltas_01,
    double Rcr,
    double Rca,
    const Tensor& EtaR,
    const Tensor& ShfR,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  Tensor centers = torch::cat({pairs[0], pairs[1]});
  Tensor sides = torch::cat({pairs[1], pairs[0]});
  Tensor deltas = torch::cat({deltas_01, -deltas_01});
  return compute_aev(
      species, centers, sides, deltas, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cos_cutoff);
}

Tensor run_aten(
    const Tensor& coordinates,
    const Tensor& species,
    double Rcr,
    double Rca,
    const Tensor& EtaR,
    const Tensor& ShfR,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  int64_t num_molecules = species.size(0);
  int64_t num_atoms = species.size(1);
  auto long_opts = species.options().dtype(torch::kLong);
  Tensor species_flat = species.reshape(-1).to(torch::kLong);
  Tensor coords_flat = coordinates.reshape({-1, 3});

  Tensor offsets = torch::arange(num_molecules, long_opts) * num_atoms;
  Tensor pairs = torch::triu_indices(num_atoms, num_atoms, 1, long_opts);
  pairs = (pairs.view({2, 1, -1}) + offsets.view({1, -1, 1})).view({2, -1});
  Tensor pair_is_real = (species_flat.index_select(0, pairs[0]) != -1) & (species_flat.index_select(0, pairs[1]) != -1);
  pairs = pairs.index({Slice(), pair_is_real});

  Tensor deltas_01 = coords_flat.index_select(0, pairs[1]) - coords_flat.index_select(0, pairs[0]);
  return compute_aev_from_half_pairs(
      species, pairs, deltas_01, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cos_cutoff);
}

// diff_vectors are (coords[atomIJ[0]] - coords[atomIJ[1]] + shifts). Distances are recomputed
// from diff_vectors, so the full gradient wrt the pairs flows to diff_vectors, as in the CUDA impl
Tensor run_with_half_nbrlist_aten(
    const Tensor& species,
    const Tensor& atomIJ,
    const Tensor& diff_vectors,
    const Tensor& distances,
    double Rcr,
    double Rca,
    const Tensor& EtaR,
    const Tensor& ShfR,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  TORCH_CHECK(atomIJ.size(0) == 2, "atomIJ's shape should be (2, P)");
  Tensor pairs = atomIJ.to(torch::kLong);
  Tensor deltas_01 = -diff_vectors;
  return compute_aev_from_half_pairs(
      species, pairs, deltas_01, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cos_cutoff);
}

Tensor run_with_full_nbrlist_aten(
    const Tensor& coordinates,
    const Tensor& species,
    const Tensor& atomI,
    const Tensor& atomJ,
    const Tensor& numJPerI,
    double Rcr,
    double Rca,
    const Tensor& EtaR,
    const Tensor& ShfR,
    const Tensor& EtaA,
    const Tensor& Zeta,
    const Tensor& ShfA,
    const Tensor& ShfZ,
    int64_t num_species,
    bool use_cos_cutoff) {
  TORCH_CHECK(species.size(0) == 1, "run_with_full_nbrlist currently only supports a single molecule");
  TORCH_CHECK(atomI.size(0) == numJPerI.size(0), "atomI and numJPerI should have the same shape");
  Tensor centers = torch::repeat_interleave(atomI.to(torch::kLong), numJPerI.to(torch::kLong));
  Tensor sides = atomJ.to(torch::kLong);
  Tensor coords_flat = coordinates.reshape({-1, 3});
  Tensor deltas = coords_flat.index_select(0, sides) - coords_flat.index_select(0, centers);
  return compute_aev(
      species, centers, sides, deltas, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cos_cutoff);
}

// The ATen implementation is differentiable, so it is also registered to AutogradCPU,
// where autograd tracks the ops it calls
TORCH_LIBRARY_IMPL(cuaev, CPU, m) {
  m.impl("run", run_aten);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_aten);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_aten);
}

TORCH_LIBRARY_IMPL(cuaev, AutogradCPU, m) {
  m.impl("run", run_aten);
  m.impl("run_with_half_nbrlist", run_with_half_nbrlist_aten);
  m.impl("run_with_full_nbrlist", run_with_full_nbrlist_aten);
}
