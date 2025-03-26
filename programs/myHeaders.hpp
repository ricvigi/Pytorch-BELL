#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP
#include <torch/torch.h>
#include <cstdio>
#include <vector>
#include <cmath>

static inline void printTensor(const torch::Tensor &A, int x, int y)
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

/* ATTENTION: Find a better way to implement this algorithm. NOTE: keep in mind that we need to return a sorted list.
 * The algorithm returns all divisors of x up to x / 2 */
static inline std::vector<int> findDivisors(int x) {
    std::vector<int> divisors;
    int i = 2;
    while (i <= (x / 2)) /* With normal iteration the result array is naturally sorted */
    {
      if (x % i == 0)
      {
        divisors.push_back(i);
      }
      i++;
    }
    return divisors; /* ATTENTION: Must return a sorted array */
}

#endif
