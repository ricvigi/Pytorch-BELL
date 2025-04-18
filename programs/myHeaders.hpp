#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP
#include <torch/torch.h>
#include <cstdio>
#include <cmath>

int getBellParams(torch::Tensor& A,         /* in */
                  int x,                    /* in */
                  int y,                    /* in */
                  int& ellBlockSize,        /* out */
                  int& ellCols,             /* out */
                  int*& ellColInd,          /* out */
                  float*& ellValue);        /* out */

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
      printf("%7.4f ", A.index({i, j}).item<float>());
    }
    printf("\n");
  }
}

/**
 * @brief Determines whether a number is prime or not
 *
 * @param x The number that we need to check
 *
 * @return 1 if x is prime, 0 otherwise
 */
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


/**
 * @brief Prints a 2D matrix of unspecified (hopefully) numeric type
 *
 * @param M The matrix
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return void
 */
template <typename T> /* ATTENTION: We should check that T is strictly numeric */
inline void printMat(const T* M, int rows, int cols)
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

/**
 * @brief Determines whether a number is prime or not
 *
 * @param x The number that we need to check
 *
 * @return 1 if x is prime, 0 otherwise
 */
static inline void printEllValue(float* ellValue, int rows, int cols, int kernelSize) {
  for (int i = 0; i < rows * kernelSize; ++i) {
    for (int j = 0; j < cols * kernelSize; ++j) {
      int blockRow = i / kernelSize;
      int blockCol = j / kernelSize;
      int bi = i % kernelSize;
      int bj = j % kernelSize;
      int index = ((blockRow * cols + blockCol) * kernelSize + bi) * kernelSize + bj;
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
