#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void custom_add_kernel(const float* x, const float* y, float* out, int64_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + y[idx];
  }
}

}  // namespace

torch::Tensor custom_add_cuda(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  auto out = torch::zeros_like(x);
  const int64_t n = x.numel();

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  custom_add_kernel<<<blocks, threads>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);

  return out;
}
