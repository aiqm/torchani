#pragma once

// include cub in a safe manner, see:
// https://github.com/pytorch/pytorch/pull/55292
#undef CUB_NS_POSTFIX // undef to avoid redefinition warnings
#undef CUB_NS_PREFIX
#define CUB_NS_PREFIX namespace cuaev {
#define CUB_NS_POSTFIX }
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX

template <typename DataT>
void cubScan(const DataT* d_in, DataT* d_out, int num_items, cudaStream_t stream) {
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cuaev::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run exclusive prefix sum
  cuaev::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
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
  cuaev::cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run encoding
  cuaev::cub::DeviceRunLengthEncode::Encode(
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
  cuaev::cub::DeviceSelect::If(
      d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run selection
  cuaev::cub::DeviceSelect::If(
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
  cuaev::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  // Allocate temporary storage
  auto buffer_tmp = allocator.allocate(temp_storage_bytes);
  d_temp_storage = buffer_tmp.get();

  // Run min-reduction
  cuaev::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);

  int maxVal = 0;
  cudaMemcpyAsync(&maxVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  return maxVal;
}
