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

#include <ATen/cuda/Exceptions.h>

// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                   \
  do {                                                           \
    size_t temp_storage_bytes = 0;                               \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);              \
    auto temp_storage = allocator->allocate(temp_storage_bytes); \
    func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);   \
    AT_CUDA_CHECK(cudaGetLastError());                           \
  } while (false)

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
