#include <ATen/ATen.h>

int main(int argc, char** argv)
{
  at::Tensor a = at::ones({2,2}, at::kInt);
  at::Tensor b = at::randn({2,2});
  // std::cout << a << std::endl;
  // std::cout << b << std::endl;
  auto c = a+b.to(at::kInt);
  std::cout << a,b,c << std::endl;
  return 0;
}
