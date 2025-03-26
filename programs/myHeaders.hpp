#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP

static inline void print_tensor(const torch::Tensor &A, int x, int y)
{
  for (int i = 0; i < x; i++)
  {
    for (int j = 0; j < y; j++)
    {
      printf("%f ", A.index({i, j}).item<float>());
    }
    printf("\n");
  }
}

#endif
