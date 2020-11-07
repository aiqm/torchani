#include <cub/cub.cuh>
#include <string>
#include <torch/extension.h>

__global__ void kernel() { printf("Hello World!"); }

std::string say_hello() {
  kernel<<<1, 1>>>();
  return "Hello World!!!";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuComputeAEV", &say_hello, "Hello World");
}
