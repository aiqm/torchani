#include <torch/extension.h>
#include <tuple>
#include <vector>

// using namespace at;
using torch::Tensor;
using torch::autograd::tensor_list;

Tensor map_to_central(const Tensor& coordinates, const Tensor& cell, const Tensor& pbc) {
  Tensor inv_cell = torch::inverse(cell);
  Tensor coordinates_cell = torch::matmul(coordinates, inv_cell);
  coordinates_cell -= coordinates_cell.floor() * pbc;
  Tensor mapped_coordinates = torch::matmul(coordinates_cell, cell);
  return mapped_coordinates;
}

Tensor setup_grid(const Tensor& cell, double cutoff, int64_t buckets_per_cutoff = 1, double extra_space = 1.0e-5) {
  Tensor spherical_factor = cell.new_ones({3});
  Tensor bucket_length_lower_bound = (spherical_factor * cutoff / buckets_per_cutoff) + extra_space;
  Tensor cell_lengths = torch::norm(cell, 2, 0);
  Tensor grid_shape = torch::floor_divide(cell_lengths, bucket_length_lower_bound).to(torch::kLong);
  return grid_shape;
}

// TODO: Throw c10 errors?
void _validate_inputs(
    double cutoff,
    const Tensor& species,
    const Tensor& coords,
    const c10::optional<Tensor>& cell,
    const c10::optional<Tensor>& pbc) {
  if (cutoff <= 0.0f) {
    throw std::invalid_argument("Cutoff must be a strictly positive float");
  }
  if (coords.size(0) != 1) {
    throw std::invalid_argument("This neighborlist doesn't support batches");
  }
  if (pbc.has_value()) {
    if (!pbc.value().any().item<bool>()) {
      throw std::invalid_argument(
          "pbc = torch.tensor([False, False, False]) is not supported anymore"
          " please use pbc = None");
    }
    if (!cell.has_value()) {
      throw std::invalid_argument("If pbc is not None, cell should be present");
    }
    if (!pbc.value().all().item<bool>()) {
      throw std::invalid_argument("This neighborlist doesn't support PBC only in some directions");
    }
  } else {
    // throw std::invalid_argument("Currently non-pbc is broken in FastCellList");
    if (cell.has_value()) {
      throw std::invalid_argument("Cell is not supported if not using pbc");
    }
  }
}

tensor_list narrow_down(
    double cutoff,
    const Tensor& elem_idxs,
    const Tensor& coords,
    Tensor neighbor_idxs,
    c10::optional<Tensor> shifts = c10::nullopt) {
  Tensor mask = elem_idxs == -1;
  if (mask.any().item<bool>()) {
    mask = mask.view(-1).index_select(0, neighbor_idxs.view(-1)).view({2, -1});
    Tensor non_dummy_pairs = (~torch::any(mask, 0)).nonzero().flatten();
    neighbor_idxs = neighbor_idxs.index_select(1, non_dummy_pairs);
    if (shifts.has_value()) {
      shifts = shifts.value().index_select(0, non_dummy_pairs);
    }
  }

  Tensor coords_flat = coords.view({-1, 3});
  Tensor _coords = coords_flat.detach();
  Tensor _coords0 = _coords.index_select(0, neighbor_idxs[0]);
  Tensor _coords1 = _coords.index_select(0, neighbor_idxs[1]);
  Tensor _diff_vectors = _coords0 - _coords1;
  if (shifts.has_value()) {
    _diff_vectors += shifts.value();
  }
  Tensor in_cutoff = (_diff_vectors.norm(2, -1) <= cutoff).nonzero().flatten();

  neighbor_idxs = neighbor_idxs.index_select(1, in_cutoff);
  Tensor coords0 = coords_flat.index_select(0, neighbor_idxs[0]);
  Tensor coords1 = coords_flat.index_select(0, neighbor_idxs[1]);
  Tensor diff_vectors = coords0 - coords1;
  if (shifts.has_value()) {
    diff_vectors += shifts.value().index_select(0, in_cutoff);
  }
  Tensor distances = diff_vectors.norm(2, -1);
  return {neighbor_idxs, distances, diff_vectors};
}

std::tuple<Tensor, Tensor> compute_bounding_cell(
    const Tensor& coords,
    double eps = 1.0e-3,
    bool displace = true,
    bool square = false) {
  Tensor min_ = std::get<0>(torch::min(coords.view({-1, 3}), 0)) - eps;
  Tensor max_ = std::get<0>(torch::max(coords.view({-1, 3}), 0)) + eps;
  Tensor largest_dist = max_ - min_;
  Tensor cell;
  if (square) {
    cell = torch::eye(3, coords.device()) * largest_dist.max();
  } else {
    cell = torch::eye(3, coords.device()) * largest_dist;
  }
  if (displace) {
    return {coords - min_, cell};
  }
  return {coords, cell};
}

Tensor coords_to_grid_idx3(const Tensor& coords, const Tensor& cell, const Tensor& grid_shape) {
  Tensor fractional_coords = torch::remainder(torch::matmul(coords, cell.inverse()), 1.0);
  return (fractional_coords * grid_shape).floor().to(torch::kLong);
}

Tensor flatten_idx3(const Tensor& idx3, const Tensor& grid_shape) {
  Tensor grid_factors = grid_shape.clone();
  grid_factors[0] = grid_shape[1] * grid_shape[2];
  grid_factors[1] = grid_shape[2];
  grid_factors[2] = 1;
  return (idx3 * grid_factors).sum(-1);
}

std::tuple<Tensor, Tensor> atom_image_converters(const Tensor& grid_idx) {
  Tensor image_to_atom = torch::argsort(grid_idx.view(-1));
  Tensor atom_to_image = torch::argsort(image_to_atom);
  return {atom_to_image, image_to_atom};
}

Tensor cumsum_from_zero(const Tensor& input) {
  Tensor out = torch::zeros_like(input);
  out.slice(0, 1, out.size(0)) = input.cumsum(0).slice(0, 0, input.size(0) - 1);
  return out;
}

std::tuple<Tensor, Tensor> count_atoms_in_buckets(const Tensor& atom_grid_idx, const Tensor& grid_shape) {
  Tensor flattened_idx = atom_grid_idx.view(-1);
  Tensor count_in_grid = at::bincount(flattened_idx, /*weights*/ {}, /*minlength=*/grid_shape.prod().item<int64_t>());
  Tensor cumulative_count = cumsum_from_zero(count_in_grid);
  return {count_in_grid, cumulative_count};
}

Tensor nonzero_in_chunks(const Tensor& tensor, int64_t chunk_size = 2147483647) {
  Tensor flat_tensor = tensor.view({-1});
  int64_t num_elements = flat_tensor.numel();
  int64_t num_splits = std::ceil(static_cast<double>(num_elements) / chunk_size);
  if (num_splits <= 1) {
    return flat_tensor.nonzero().view(-1);
  }

  int64_t offset = 0;
  std::vector<Tensor> nonzero_chunks;
  for (const auto& chunk : torch::chunk(flat_tensor, num_splits)) {
    nonzero_chunks.push_back(chunk.nonzero() + offset);
    offset += chunk.size(0);
  }
  return torch::cat(nonzero_chunks).view(-1);
}

Tensor fast_masked_select(const Tensor& x, const Tensor& mask, int64_t idx) {
  return x.index_select(idx, nonzero_in_chunks(mask));
}

Tensor image_pairs_within(const Tensor& count_in_grid, const Tensor& cumcount_in_grid, int64_t count_in_grid_max) {
  torch::Device device = count_in_grid.device();

  Tensor haspairs_idx_to_grid_idx = (count_in_grid > 1).nonzero().view(-1);
  Tensor count_in_haspairs = count_in_grid.index_select(0, haspairs_idx_to_grid_idx);
  Tensor cumcount_in_haspairs = cumcount_in_grid.index_select(0, haspairs_idx_to_grid_idx);

  Tensor image_pairs_in_fullest_bucket = torch::tril_indices(
      count_in_grid_max, count_in_grid_max, -1, torch::TensorOptions().dtype(torch::kLong).device(device));
  Tensor _image_pairs_within = image_pairs_in_fullest_bucket.view({2, 1, -1}) + cumcount_in_haspairs.view({1, -1, 1});
  _image_pairs_within = _image_pairs_within.view({2, -1});

  Tensor paircount_in_haspairs = (count_in_haspairs * (count_in_haspairs - 1)) / 2;
  Tensor mask = torch::arange(0, image_pairs_in_fullest_bucket.size(1), device = device);
  mask = mask.view({1, -1}) < paircount_in_haspairs.view({-1, 1});

  return fast_masked_select(_image_pairs_within, mask, 1);
}

std::tuple<Tensor, Tensor> lower_image_pairs_between(
    const Tensor& count_in_atom_surround, // shape (C, A, N=13)
    const Tensor& cumcount_in_atom_surround, // shape (C, A, N=13)
    Tensor shift_idxs_between, // shape (C, A, N=13, 3)
    int64_t count_in_grid_max // scalar
) {
  torch::Device device = count_in_atom_surround.device();
  int64_t mols = count_in_atom_surround.size(0);
  int64_t atoms = count_in_atom_surround.size(1);
  int64_t neighbors = count_in_atom_surround.size(2);

  // Padded atom neighbors: shape (C, A, N=13, c-max)
  Tensor padded_atom_neighbors = torch::arange(0, count_in_grid_max, device = device).view({1, 1, 1, -1});
  padded_atom_neighbors = padded_atom_neighbors.repeat({mols, atoms, neighbors, 1});

  // Create mask for unpadded neighbors
  Tensor mask = padded_atom_neighbors < count_in_atom_surround.unsqueeze(-1);
  padded_atom_neighbors += cumcount_in_atom_surround.unsqueeze(-1);

  // Pad the shift indices: shape (C, A, N=13, c-max, 3)
  shift_idxs_between = shift_idxs_between.unsqueeze(-2).repeat({1, 1, 1, count_in_grid_max, 1});

  // Apply mask to get the lower between values and shift indices
  Tensor lower_between = fast_masked_select(padded_atom_neighbors.view(-1), mask, 0);
  shift_idxs_between = fast_masked_select(shift_idxs_between.view({-1, 3}), mask, 0);
  return {lower_between, shift_idxs_between};
}

std::tuple<Tensor, Tensor> _cell_list(const Tensor& grid_shape, const Tensor& coords, const Tensor& cell) {
  Tensor atom_grid_idx3 = coords_to_grid_idx3(coords, cell, grid_shape); // Checked
  Tensor atom_grid_idx = flatten_idx3(atom_grid_idx3, grid_shape); // Checked

  Tensor count_in_grid, cumcount_in_grid;
  std::tie(count_in_grid, cumcount_in_grid) = count_atoms_in_buckets(atom_grid_idx, grid_shape); // Checked
  int64_t count_in_grid_max = count_in_grid.max().item<int64_t>();

  Tensor _image_pairs_within = image_pairs_within(count_in_grid, cumcount_in_grid, count_in_grid_max); // Checked

  Tensor offset_idx3 = torch::tensor(
      {{{{-1, 0, 0},
         {-1, -1, 0},
         {0, -1, 0},
         {1, -1, 0},
         {-1, 1, -1},
         {0, 1, -1},
         {1, 1, -1},
         {-1, 0, -1},
         {0, 0, -1},
         {1, 0, -1},
         {-1, -1, -1},
         {0, -1, -1},
         {1, -1, -1}}}},
      torch::TensorOptions().dtype(torch::kLong).device(coords.device()));
  Tensor atom_surr_idx3 = atom_grid_idx3.unsqueeze(-2) + offset_idx3;
  Tensor shift_idxs_between = -torch::floor_divide(atom_surr_idx3, grid_shape);
  Tensor atom_surr_idx = flatten_idx3(atom_surr_idx3 % grid_shape, grid_shape); // Checked

  Tensor count_in_atom_surround = count_in_grid.index({atom_surr_idx});
  Tensor cumcount_in_atom_surround = cumcount_in_grid.index({atom_surr_idx});

  Tensor lower_between; // Not checked (seems good)
  std::tie(lower_between, shift_idxs_between) = lower_image_pairs_between(
      count_in_atom_surround, cumcount_in_atom_surround, shift_idxs_between, count_in_grid_max);

  Tensor total_count_in_atom_surround = count_in_atom_surround.sum(-1).view(-1);

  Tensor atom_to_image, image_to_atom; // Not checked (seems good)
  std::tie(atom_to_image, image_to_atom) = atom_image_converters(atom_grid_idx);

  Tensor upper_between = torch::repeat_interleave(atom_to_image, total_count_in_atom_surround);
  Tensor _image_pairs_between = torch::stack({upper_between, lower_between}, 0);

  Tensor shift_idxs_within = torch::zeros({_image_pairs_within.size(1), 3}, grid_shape.options().dtype(torch::kLong));
  Tensor shift_idxs = torch::cat({shift_idxs_between, shift_idxs_within}, 0);
  Tensor image_pairs = torch::cat({_image_pairs_between, _image_pairs_within}, 1);
  Tensor neighbor_idxs = image_to_atom.index({image_pairs});
  return {neighbor_idxs, shift_idxs};
}

class CellListFunction : public torch::autograd::Function<CellListFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      double cutoff,
      const Tensor& species,
      const Tensor& coords,
      const c10::optional<Tensor>& cell,
      const c10::optional<Tensor>& pbc) {
    _validate_inputs(cutoff, species, coords, cell, pbc);
    Tensor displ_coords;
    Tensor out_cell;
    if (pbc.has_value()) {
      TORCH_CHECK(cell.has_value(), "Cell must be provided when PBC is enabled");
      displ_coords = coords.detach();
      out_cell = cell.value();
    } else {
      std::tie(displ_coords, out_cell) = compute_bounding_cell(
          coords.detach(),
          /* eps */ (2 * cutoff + 1.0e-3),
          /* displace */ true,
          /* square */ false);
    }
    Tensor grid_shape = setup_grid(out_cell, cutoff);
    if (pbc.has_value()) {
      if ((grid_shape == 0).any().item<bool>()) {
        throw std::runtime_error("Cell is too small to perform PBC calculations");
      }
    } else {
      grid_shape = torch::max(grid_shape, torch::ones_like(grid_shape));
    }

    Tensor neighbor_idxs, shift_idxs;
    std::tie(neighbor_idxs, shift_idxs) = _cell_list(grid_shape, displ_coords.detach(), out_cell);

    tensor_list outs;
    if (pbc.has_value()) {
      Tensor shifts = torch::matmul(shift_idxs.to(out_cell.dtype()), out_cell);
      Tensor map_coords = map_to_central(coords, out_cell.detach(), pbc.value());
      outs = narrow_down(cutoff, species, map_coords, neighbor_idxs, shifts);
    } else {
      outs = narrow_down(cutoff, species, coords, neighbor_idxs);
    }
    ctx->save_for_backward({
        coords,
        outs[0], // neighbor_idxs
        outs[1], // distances
        outs[2], // diff_vectors
    });
    return outs;
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    Tensor coords = saved[0];
    Tensor neighbor_idxs = saved[1];
    Tensor distances = saved[2];
    Tensor diff_vectors = saved[3];

    // grad outputs has gradients wrt neighbor_idxs, distances, diff_vectors
    // but gradient wrt species is not needed (long tensor)
    Tensor grad_distances = grad_outputs[1];
    Tensor grad_diff_vectors = grad_outputs[2];
    Tensor norm_diff_vectors = diff_vectors / distances.clamp(1.0e-12).unsqueeze(-1);

    // Add gradient wrt distances and diff vectors
    Tensor grad_ij = grad_distances.unsqueeze(-1) * norm_diff_vectors + grad_diff_vectors;

    // The gradient in ij is -1 * the gradient in ji
    Tensor grad_coords = torch::zeros_like(coords).squeeze(0);
    grad_coords.index_add_(0, neighbor_idxs[0], grad_ij);
    grad_coords.index_add_(0, neighbor_idxs[1], -grad_ij);
    return {Tensor(), Tensor(), grad_coords.unsqueeze(0), Tensor(), Tensor()};
  }
};

std::tuple<Tensor, Tensor, Tensor> cell_list(
    double cutoff,
    const Tensor& species,
    const Tensor& coords,
    const c10::optional<Tensor>& cell,
    const c10::optional<Tensor>& pbc) {
  tensor_list output = CellListFunction::apply(cutoff, species, coords, cell, pbc);
  return {output[0], output[1], output[2]};
}
TORCH_LIBRARY(cell_list, m) {
  m.def(
      "cell_list(float cutoff, Tensor species, Tensor coords, Tensor? cell, Tensor? pbc) -> (Tensor, Tensor, Tensor)",
      &cell_list);
  m.def("coords_to_grid_idx3(Tensor coords, Tensor cell, Tensor grid_shape) -> Tensor", &coords_to_grid_idx3);
  m.def(
      "setup_grid(Tensor cell, float cutoff, int buckets_per_cutoff=1, float extra_space=1e-5) -> Tensor", &setup_grid);
  m.def("flatten_idx3(Tensor idx3, Tensor grid_shape) -> Tensor", &flatten_idx3);
  m.def(
      "image_pairs_within(Tensor count_in_grid, Tensor cumcount_in_grid, int count_in_grid_max) -> Tensor",
      &image_pairs_within);
  m.def("count_atoms_in_buckets(Tensor atom_grid_idx, Tensor grid_shape) -> (Tensor, Tensor)", &count_atoms_in_buckets);
}
