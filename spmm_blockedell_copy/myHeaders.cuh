#ifndef MY_HEADERS_HPP
#define MY_HEADERS_HPP
#include <torch/torch.h>
#include <cusparse.h>         // cusparseSpMM
#include <cuda_fp16.h>        // data types
#include <type_traits>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include <torch/torch.h>
#include <omp.h>
#include <math.h>
#include <cstdio>
#include <cmath>
#include <ATen/ATen.h>


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

using at::Half;

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

template<>
struct cuda_dtype<__half>
{
    static constexpr cudaDataType_t val = CUDA_R_16F;
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

// template <>
// struct scalar_type<at::Half>
// {
//   static constexpr torch::ScalarType val = torch::kFloat16;
// };


template <typename T>
int getBellParams(torch::Tensor& A,         /* in */
                  int x,                    /* in */
                  int y,                    /* in */
                  int& ellBlockSize,        /* out */
                  int& ellCols,             /* out */
                  int*& ellColInd,          /* out */
                  T*& ellValue);            /* out */


/**
 * @brief converts matrix A (pointer) into blockedell format and returns a descriptor object of the blockedell format
 */
template <typename T>
__host__ int convert_to_blockedell(torch::Tensor A  /* in */, cusparseDnMatDescr_t &matA ,cusparseSpMatDescr_t &spA /* out */, int *dA_columns, T *dA_values, T *dA_dense, int *ellBlockSize, int *ellCols, int *ellColInd, T *ellValue)
{
  unsigned int A_rows = A.size(0);
  unsigned int A_cols = A.size(1);
  printf("%d %d\n", A_rows, A_cols);
  unsigned int lda = A_cols;

  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  T *hA = A.contiguous().data_ptr<T>();

  // Get the ellColInd array for matrix A
  int err;


  err = getBellParams<T>(A, A_rows, A_cols, *ellBlockSize, *ellCols, ellColInd, ellValue);
  if (err != 0)
  {
    printf("Error code %d, exiting!\n", err);
    fflush(stdout);
    return err;
  }

  // ATTENTION: ellCols is usually considered to be the number of columns in ell format, NOT the number of blocks (of the ell format).
  *ellCols = (*ellBlockSize) * (*ellCols);
  // Device memory management
  // printf("ellCols: %d, n_non_zeroes: %d\n", ellCols, n_non_zeroes);
  int ellColInd_size = A_rows * (*ellCols);
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  CHECK_CUDA(cudaMallocAsync((void**) &dA_dense, A_rows * A_cols * sizeof(float), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_columns, ellColInd_size * sizeof(int), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_values, A_rows * (*ellCols) * sizeof(float), stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_dense, hA, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_columns, ellColInd, ellColInd_size * sizeof(int), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemsetAsync(dA_values, 0.0f, A_rows * (*ellCols) * sizeof(float), stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))

  /* [BEGIN] Dense to sparse conversion */
  // To create a conversion you need a dense matrix to convert it into a sparse matrix. If you want to store matrix A
  // in a sparse format, you need to convert A's dense representation to sparse!
  cusparseHandle_t conversionHandle = NULL;
  void *dBuffer    = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&conversionHandle))

  /* ATTENTION: remember that leading dimension is number of columns if we use CUSPARSE_ORDER_ROW, and vice versa */
  // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_rows, A_cols, lda, dA_dense,
                                      cuda_type, CUSPARSE_ORDER_ROW) )

  // Create sparse matrix B in Blocked ELL format
  CHECK_CUSPARSE( cusparseCreateBlockedEll(&spA, A_rows, A_cols,
                                           (*ellBlockSize), (*ellCols),
                                           dA_columns, dA_values,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           cuda_type) )

  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(conversionHandle, matA, spA,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(conversionHandle, matA, spA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_convert(conversionHandle, matA, spA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))
  /* [END] Dense to sparse conversion */


  CHECK_CUDA(cudaFree(dBuffer))
  // CHECK_CUDA(cudaFree(dA_values))
  // CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroy(conversionHandle))
  // free(ellColInd);
  // free(ellValue);
  return EXIT_SUCCESS;
}



/**
 * @brief This function execute sparse matrix-matrix multiplication and stores the result in dense matric C.
 * @note If you consider B as a m x 1 vector, it's also an implementation of sparse-vector multiplication
 */
template <typename T>
__host__ int execute_spmm(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, T alpha, T beta)
{

  cusparseHandle_t multiplicationHandle = NULL;
  void* dBufferMul = NULL;
  size_t bufferMulSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&multiplicationHandle) )

  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                  multiplicationHandle,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, spA, B, &beta, C, cuda_type,
                  CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferMulSize) )

  CHECK_CUDA( cudaMalloc(&dBufferMul, bufferMulSize) )

  // execute SpMM
  CHECK_CUSPARSE( cusparseSpMM(multiplicationHandle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, spA, B, &beta, C, cuda_type,
                               CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBufferMul) )
  CHECK_CUDA( cudaFree(dBufferMul) )
  return EXIT_SUCCESS;
}



/* Returns the number of non-zero values of matrix float *mat. rows and cols are the dimensions of *mat, and n_non_zeroes is the return value (should be initialized to 0 before calling this function). */
__host__ inline void
count_non_zeroes(float *mat, unsigned int rows, unsigned int cols, unsigned int *n_non_zeroes)
{
  const float eps = 1e-9;
  // number of non zero values in *mat

  for (unsigned int i = 0; i < rows; ++i)
  {
    for (unsigned int j = 0; j < cols; ++j)
    {
      float t = 0.0f;
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
__host__ inline void
extract_non_zeros(float *mat, unsigned int rows, unsigned int cols, float *non_zero_values)
{
  const float eps = 1e-9;
  unsigned int idx = 0;

  for (unsigned int i = 0; i < rows; ++i)
  {
    for (unsigned int j = 0; j < cols; ++j)
    {
      float t = 0.0f;
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
