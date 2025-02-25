#include <torch/torch.h>
#include <iostream>
#include <string>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}


int main()
{
    torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
    torch::Tensor b = torch::randn({2, 2});
    double c = 1.22;
    torch::Tensor result = mymuladd_cpu(a,b,c);
    std::cout << result << std::endl;
    return 0;
}
