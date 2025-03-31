#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP
#include <torch/torch.h>
#include <cstdio>
#include <cmath>

/**
 * @brief Prints the values stored in a two dimensional tensor of size (x,y)
 *
 * @param &A Reference to the tensor we wish to print
 * @param x Size of the first dimension of A
 * @param y Size of the second dimension of A
 *
 * @return void
 */
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

static inline int isPrime(const int x)
{
  int root = (int) sqrt(x);
  int prime = 1;
  for (int i = 2; i <= root; i++)
  {
    if ((x % i) == 0)
    {
      prime = 0;
      break;
    }
  }
  return prime;
}

template <typename T> /* ATTENTION: We should check that T is strictly numeric */
static inline void printMat(const T* M, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      std::cout << M[i * cols + j] << " ";
    }
    printf("\n");
  }
}

static inline void printEllValue(float* ellValue, int rows, int cols, int res) {
  for (int i = 0; i < rows * res; ++i) {
    for (int j = 0; j < cols * res; ++j) {
      int blockRow = i / res;
      int blockCol = j / res;
      int bi = i % res;
      int bj = j % res;
      int index = ((blockRow * cols + blockCol) * res + bi) * res + bj;
      printf("%7.4f ", ellValue[index]);
    }
    printf("\n");
  }
}

/* ATTENTION: Find a faster way (if possible) to implement this algorithm. NOTE: keep in mind that we need to return a sorted list.
 * The algorithm returns all divisors of x up to x / 2 */
/**
 * @brief Finds all the divisors of a number x, up to x / 2
 *
 * @param x The number we wish to find the divisors of
 * @param **divisors A double pointer to an (initialized) array of integers of one element.
 *
 * @return The size of the array of integers that stores all the found divisors of x
 */
static inline int findDivisors(int x, int*& divisors)
{
  int size = 0;
  int i;
# pragma omp parallel for schedule(static, 1)
  for (i = 2; i <= (x / 2); i++) /* With normal iteration the result array is naturally sorted */
  {
    if (x % i == 0)
    {
#     pragma omp critical
      {
        size += 1;
        divisors = (int*) realloc(divisors, sizeof(int)*size);
      }
      divisors[size - 1] = i;
    }
  }
  return size;
}

#endif
