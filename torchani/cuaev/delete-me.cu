// This file only exists temporarily for validating the clang-format CI
// This file should be deleted when https://github.com/aiqm/torchani/pull/516
// or https://github.com/aiqm/torchani/pull/521 is merged.

#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_gpu(void) {
  printf("Hello World! from thread [%d,%d] \
        From device\n",
         threadIdx.x, blockIdx.x);
}

int main(void) {
  printf("Hello World from host!\n");
  print_from_gpu<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
