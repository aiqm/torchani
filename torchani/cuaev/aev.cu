#include <string>
#include <torch/extension.h>
#include <cub/cub.cuh>

__global__ void kernel() {
  printf("Hello World!");
}

std::string say_hello() {
  kernel<<<1,1>>>();
  return "Hello World!!!";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuComputeAEV", &say_hello, "Hello World");
}

bool is_installed() {
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_installed", &is_installed, "Check if cuaev is installed");
}
