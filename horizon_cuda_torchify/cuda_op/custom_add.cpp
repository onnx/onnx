#include <torch/extension.h>

torch::Tensor custom_add_cuda(torch::Tensor x, torch::Tensor y);

torch::Tensor custom_add(torch::Tensor x, torch::Tensor y) {
  if (x.is_cuda()) {
    return custom_add_cuda(x, y);
  }
  return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_add", &custom_add, "Custom Add (CUDA + CPU fallback)");
}
