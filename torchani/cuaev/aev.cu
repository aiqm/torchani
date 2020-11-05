#include <string>
#include <torch/extension.h>
#include <cub/cub.cuh>

__global__ void kernel() {
  printf("Hello World!");
}

std::string say_hello() {
  return "Hello World!!!";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuComputeAEV", &say_hello, "Hello World");
}
