#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

// Handle the temporary storage and 'twice' calls for cub API
// C10_CUDA_CHECK defined in c10/cuda/Exception.h
#define CUB_WRAPPER(func, ...)                                   \
  do {                                                           \
    size_t temp_storage_bytes = 0;                               \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);              \
    auto temp_storage = allocator->allocate(temp_storage_bytes); \
    func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);   \
    C10_CUDA_CHECK(cudaGetLastError());                          \
  } while (false)

inline int get_num_bits(uint64_t max_key) {
  int num_bits = 1;
  while (max_key > 1) {
    max_key >>= 1;
    num_bits++;
  }
  return num_bits;
}

template <typename DataT>
void cubScan(const DataT* d_in, DataT* d_out, int num_items, cudaStream_t stream) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  CUB_WRAPPER(cuaev::cub::DeviceScan::ExclusiveSum, d_in, d_out, num_items, stream);
}

template <typename DataT, typename LambdaOpT>
int cubDeviceSelectIf(
    const DataT* d_in,
    DataT* d_out,
    int num_items,
    LambdaOpT select_op,
    cudaStream_t stream,
    bool sync = true) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  int* d_num_selected_out = (int*)buffer_count.get();

  CUB_WRAPPER(cuaev::cub::DeviceSelect::If, d_in, d_out, d_num_selected_out, num_items, select_op, stream);

  // TODO copy num_selected to host, this part is slow
  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_selected_out, sizeof(int), cudaMemcpyDefault, stream);
  if (sync)
    cudaStreamSynchronize(stream);
  return num_selected;
}

template <typename DataT>
void cubDeviceSelectFlagged(const DataT* d_in, DataT* d_out, int num_items, char* d_flags, cudaStream_t stream) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  int* d_num_selected_out = (int*)buffer_count.get();
  CUB_WRAPPER(cuaev::cub::DeviceSelect::Flagged, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

template <typename DataT, typename IndexT>
int cubEncode(const DataT* d_in, DataT* d_unique_out, IndexT* d_counts_out, int num_items, cudaStream_t stream) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  int* d_num_runs_out = (int*)buffer_count.get();

  CUB_WRAPPER(
      cuaev::cub::DeviceRunLengthEncode::Encode, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);

  int num_unique = 0;
  cudaMemcpyAsync(&num_unique, d_num_runs_out, sizeof(int), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  return num_unique;
}

template <typename KeyT, typename ValueT>
void cubSortPairs(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    int num_items,
    int max_key,
    cudaStream_t stream) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  int nbits = get_num_bits(max_key);
  CUB_WRAPPER(
      cuaev::cub::DeviceRadixSort::SortPairs,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      0,
      nbits,
      stream);
}

template <typename DataT>
DataT cubMax(const DataT* d_in, int num_items, cudaStream_t stream, bool sync = true) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  DataT* d_out = (DataT*)buffer_count.get();

  CUB_WRAPPER(cuaev::cub::DeviceReduce::Max, d_in, d_out, num_items, stream);

  DataT maxVal = 0;
  cudaMemcpyAsync(&maxVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  if (sync)
    cudaStreamSynchronize(stream);
  return maxVal;
}

template <typename DataT>
DataT cubSum(const DataT* d_in, int num_items, cudaStream_t stream, bool sync = true) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  DataT* d_out = (DataT*)buffer_count.get();

  CUB_WRAPPER(cuaev::cub::DeviceReduce::Sum, d_in, d_out, num_items, stream);

  DataT sumVal = 0;
  cudaMemcpyAsync(&sumVal, d_out, sizeof(DataT), cudaMemcpyDefault, stream);
  if (sync)
    cudaStreamSynchronize(stream);
  return sumVal;
}
