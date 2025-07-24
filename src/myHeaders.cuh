#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP
#include <torch/torch.h>
#include <cusparse.h>         // cusparseSpMM
#include <cuda_fp16.h>        // data types
#include <type_traits>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include <iostream>
#include <torch/torch.h>
#include <omp.h>
#include <math.h>
#include <cstdio>
#include <cmath>
#include <cstdint>         // int8_t


#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                       \
{                                                                              \
  cudaError_t status = (func);                                               \
  if (status != cudaSuccess) {                                               \
    std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
    __LINE__, cudaGetErrorString(status), status);                  \
    return EXIT_FAILURE;                                                   \
  }                                                                          \
}
#endif

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
  cusparseStatus_t status = (func);                                          \
  if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
    std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
    __LINE__, cusparseGetErrorString(status), status);              \
    return EXIT_FAILURE;                                                   \
  }                                                                          \
}
#endif

extern int PRINT_DEBUG;

const int EXIT_UNSUPPORTED = 2;

template <typename T> int run (int argc, char **argv);

template<typename T>
struct cuda_dtype;

template<>
struct cuda_dtype<float>
{
    static constexpr cudaDataType_t val = CUDA_R_32F;
};

template<>
struct cuda_dtype<double>
{
    static constexpr cudaDataType_t val = CUDA_R_64F;
};

template<> // TODO: Find a way to have half precision tensors
struct cuda_dtype<int8_t>
{
    static constexpr cudaDataType_t val = CUDA_R_8I;
};
template <>
struct cuda_dtype<int>
{
  static constexpr cudaDataType_t val = CUDA_R_32I;
};

template <typename T>
struct scalar_type;

template <>
struct scalar_type<double>
{
  static constexpr torch::ScalarType val = torch::kFloat64;
};

template <>
struct scalar_type<float>
{
  static constexpr torch::ScalarType val = torch::kFloat32;
};

template <>
struct scalar_type<int8_t>
{
  static constexpr torch::ScalarType val = torch::kInt8;
};

template <>
struct scalar_type<int>
{
  static constexpr torch::ScalarType val = torch::kInt32;
};


template <typename T>
__host__ int getBellParams(torch::Tensor& A,         /* in */
                           int x,                    /* in */
                           int y,                    /* in */
                           int& ellBlockSize,        /* out */
                           int& ellCols,             /* out */
                           int*& ellColInd,          /* out */
                           T*& ellValue);            /* out */

__host__ int computeZeroBlocks (torch::Tensor &A, int rows, int cols, int kernelSize);
template <typename T> __host__ int iterativeComputeZeroBlocks (torch::Tensor &A, int rows, int cols, int kernelSize);
__host__ torch::Tensor computeEllCols (torch::Tensor& A, int rows, int cols, int kernelSize);
template <typename T> __host__ torch::Tensor iterativeComputeEllCols (torch::Tensor& A, int rows, int cols, int kernelSize);
template <typename T> __host__ void getEllColInd (torch::Tensor &bSums, int *ellColInd, int rows, int cols);
template <typename T> __host__ void getEllValues (torch::Tensor& A, T *ellValue, int *ellColInd, int rows, int cols, int ellBlockSize);
template <typename T> __host__ int getBellParams (torch::Tensor& A, int x, int y, int& ellBlockSize, int& ellCols, int*& ellColInd, T*& ellValue);

/* [BEGIN] Test functions */
template <typename T> __host__ int run(int argc, char **argv);
template <typename T> __host__ int run_int(int argc, char **argv);
/* [END] Test functions */

template <typename T>
__host__ int convert_to_blockedell(torch::Tensor &A            /* in */,
                                   cusparseDnMatDescr_t &matA  /* in */,
                                   cusparseSpMatDescr_t &spA   /* out */,
                                   int *dA_columns             /* in */,
                                   T *dA_values                /* in */,
                                   T *dA_dense                 /* in */,
                                   int *ellBlockSize           /* in */,
                                   int *ellCols                /* in */,
                                   int *ellColInd              /* in */,
                                   T *ellValue                 /* in */);



template <typename T>
__host__ int execute_spmm(cusparseSpMatDescr_t spA /* in */,
                          cusparseDnMatDescr_t B   /* in */,
                          cusparseDnMatDescr_t C   /* out */,
                          T alpha                  /* in */,
                          T beta                   /* in */);

template <typename T>
__host__ int execute_spmv(cusparseSpMatDescr_t spA,
                          cusparseDnMatDescr_t vecX,
                          cusparseDnMatDescr_t vecY,
                          T alpha,
                          T beta);


/* Returns the number of non-zero values of matrix float *mat. rows and cols are the dimensions of *mat, and n_non_zeroes is the return value (should be initialized to 0 before calling this function). */
template <typename T>
__host__ inline void count_non_zeroes(T *mat, unsigned int rows, unsigned int cols, unsigned int *n_non_zeroes)
{
  const T eps = 1e-9;
  // number of non zero values in *mat

  for (unsigned int i = 0; i < rows; ++i)
  {
    for (unsigned int j = 0; j < cols; ++j)
    {
      T t = T(0);
      t += mat[i * cols + j];
      if (fabs(t) > eps)
      {
        *n_non_zeroes += 1;
      } else
      {
        continue;
      }
    }
  }
}

/* Returns a contiguous array containing all the non zero values in *mat. The return array must be pre allocated, its size is the value returned by count_non_zeroes */
template <typename T>
__host__ inline void extract_non_zeros(T *mat, unsigned int rows, unsigned int cols, T *non_zero_values)
{
  const T eps = 1e-9;
  unsigned int idx = 0;

  for (unsigned int i = 0; i < rows; ++i)
  {
    for (unsigned int j = 0; j < cols; ++j)
    {
      T t = T(0);
      t += mat[i * cols + j];
      if (fabs(t) > eps)
      {
        non_zero_values[idx] = t;
        idx += 1;
      }
    }
  }
}


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
  for (int i = 0; i < x; ++i)
  {
    for (int j = 0; j < y; ++j)
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
  for (int i = 2; i <= root; ++i)
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
/**
 * @brief Prints a 2D matrix of unspecified (hopefully) numeric type
 *
 * @param M The matrix
 * @param rows The number of rows in the matrix
 * @param cols The number of columns in the matrix
 * @return void
 */
static inline void printMat(const T* M, int rows, int cols)
{
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
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
  for (i = 2; i <= (x / 2); ++i) /* With normal iteration the result array is naturally sorted */
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

/* This is to turn ellColInd */
template <typename T>
void rowToColMajor(const T *r_major, T *c_major, int n_rows, int n_cols)
{
  for (int i = 0; i < n_rows; ++i)
  {
    for (int j = 0; j < n_cols; ++j)
    {
      c_major[j * n_rows + i] = r_major[i * n_cols + j];
    }
  }
}


#endif
