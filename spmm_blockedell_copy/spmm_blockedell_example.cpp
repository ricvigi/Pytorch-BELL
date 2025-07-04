/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include <torch/torch.h>
#include <omp.h>
#include <math.h>
#include "myHeaders.hpp"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
  cudaError_t status = (func);                                               \
  if (status != cudaSuccess) {                                               \
    std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
    __LINE__, cudaGetErrorString(status), status);                  \
    return EXIT_FAILURE;                                                   \
  }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
  cusparseStatus_t status = (func);                                          \
  if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
    std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
    __LINE__, cusparseGetErrorString(status), status);              \
    return EXIT_FAILURE;                                                   \
  }                                                                          \
}

int PRINT_DEBUG = 0;

const int EXIT_UNSUPPORTED = 2;

/* Returns the number of non-zero values of matrix float *mat. rows and cols are the dimensions of *mat, and n_non_zeroes is the return value (should be initialized to 0 before calling this function). */
__host__ void
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
__host__ void
extract_non_zeroes(float *mat, unsigned int rows, unsigned int cols, float *non_zero_values)
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

int
main(int argc, char** argv)
{
  PRINT_DEBUG = atoi(argv[4]);
  // Host problem definition
  unsigned int A_rows = atoi(argv[1]);
  unsigned int A_cols = atoi(argv[2]);
  float threshold = atof(argv[3]);

  unsigned int B_rows = A_cols;
  unsigned int B_cols = A_cols;
  unsigned int C_rows = A_rows;
  unsigned int C_cols = B_cols;
  unsigned int lda = A_cols;
  unsigned int ldb = B_cols;
  unsigned int ldc = C_cols;

  torch::Tensor A = torch::randn({A_rows, A_cols}, torch::dtype(torch::kFloat32));
  torch::Tensor B = torch::randn({B_rows, B_cols}, torch::dtype(torch::kFloat32));
  A.masked_fill_(A < threshold, 0);
  torch::Tensor C = torch::zeros({A_rows, B_cols});

  float alpha = 1.0f;
  float beta  = 0.0f;


  std::cout << "A:\n" << A << std::endl;
  float *hA = A.contiguous().data_ptr<float>();
  float *hB = B.contiguous().data_ptr<float>();
  float *hC = C.contiguous().data_ptr<float>();

  // count how many non zero values A has
  unsigned int n_non_zeroes = 0;
  count_non_zeroes(hA, A_rows, A_cols, &n_non_zeroes);
  printf("number of non zeroes in A: %d\n", n_non_zeroes);

  // put the non zero values of A into a contiguous array
  float *non_zero_values = (float*) malloc(n_non_zeroes*sizeof(float));
  extract_non_zeroes(hA, A_rows, A_cols, non_zero_values);

  // Get the ellColInd array for matrix A
  int ellBlockSize, ellCols, err;
  int* ellColInd = nullptr;
  float* ellValue = nullptr; // We don't care about this at the moment

  err = getBellParams(A, A_rows, A_cols, ellBlockSize, ellCols, ellColInd, ellValue);
  if (err != 0)
  {
    printf("Error code %d, exiting!\n", err);
    fflush(stdout);
    return err;
  }

  // ATTENTION: ellCols is usually considered to be the number of columns in ell format, NOT the number of blocks.
  ellCols = ellBlockSize * ellCols;
  // Device memory management
  printf("ellCols: %n, n_non_zeroes: %d\n", ellCols, n_non_zeroes);
  int ellColInd_size = A_rows * ellCols;
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  int    *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA(cudaMalloc((void**) &dA_columns, ellColInd_size * sizeof(int)))
  // CHECK_CUDA(cudaMalloc((void**) &dA_values, (A_rows*ellCols*ellBlockSize) * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void**) &dA_values, A_rows * ellCols * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void**) &dB, B_rows * B_cols * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void**) &dC, C_rows * C_cols * sizeof(float)))
  CHECK_CUDA(cudaMemcpy(dA_columns, ellColInd, ellColInd_size * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, ellValue,
                        A_rows * ellCols * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, hB, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC, hC, C_rows * C_cols * sizeof(float), cudaMemcpyHostToDevice))


  printf("ellCols: %d, ellBlockSize: %d\n", ellCols, ellBlockSize);

  cusparseHandle_t     handle = NULL;
  cusparseDnMatDescr_t matA;
  cusparseSpMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle))

  /* ATTENTION: remember that leading dimension is number of columns if we use CUSPARSE_ORDER_ROW, and vice versa */
  // Create dense matrix A
  float *dA_dense;
  CHECK_CUDA(cudaMalloc((void**) &dA_dense, A_rows * A_cols * sizeof(double)))
  CHECK_CUDA(cudaMemcpy(dA_dense, hA, A_rows * A_cols * sizeof(double), cudaMemcpyHostToDevice))
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_rows, A_cols, lda, dA_dense,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )

  // Create sparse matrix B in Blocked ELL format
  CHECK_CUSPARSE( cusparseCreateBlockedEll(&matB, A_rows, A_cols,
                                           ellBlockSize, ellCols,
                                           dA_columns, dA_values,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_32F) )

  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, matA, matB,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))
  /*
   *  cusparseStatus_t
   *  cusparseDenseToSparse_bufferSize(cusparseHandle_t           handle,
   *                                   cusparseConstDnMatDescr_t  matA,  // non-const descriptor supported
   *                                   cusparseSpMatDescr_t       matB,
   *                                   cusparseDenseToSparseAlg_t alg,
   *                                   size_t*                    bufferSize)
   *  cusparseStatus_t
   *  cusparseDenseToSparse_analysis(cusparseHandle_t           handle,
   *                                 cusparseConstDnMatDescr_t  matA,  // non-const descriptor supported
   *                                 cusparseSpMatDescr_t       matB,
   *                                 cusparseDenseToSparseAlg_t alg,
   *                                 void*                      buffer)
   *  cusparseStatus_t
   *  cusparseDenseToSparse_convert(cusparseHandle_t           handle,
   *                                cusparseConstDnMatDescr_t  matA,  // non-const descriptor supported
   *                                cusparseSpMatDescr_t       matB,
   *                                cusparseDenseToSparseAlg_t alg,
   *                                void*                      buffer)
   */

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))

  float *h_ellValues = (float*) malloc(A_rows * ellCols * sizeof(float));
  CHECK_CUDA(cudaMemcpy(h_ellValues, dA_values, A_rows * ellCols * sizeof(float), cudaMemcpyDeviceToHost))
  for (unsigned int i = 0; i < A_rows; ++i)
  {
    for (unsigned int j = 0; j < ellCols; ++j)
    {
      printf("%f ", h_ellValues[i * ellCols + j]);
    }
    printf("\n");
  }

  // CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost))
  // printf("SpMM result:\n");
  // for (unsigned int i = 0; i < C_rows; ++i)
  // {
  //   for (unsigned int j = 0; j < C_cols; ++j)
  //   {
  //     printf("%f ", hC[i * C_cols + j]);
  //   }
  //   printf("\n");
  // }

  torch::Tensor res = torch::mm(A,B);
  printf("PyTorch result:\n");
  std::cout << res << std::endl;


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_dense))
  free(non_zero_values);
  free(ellColInd);
  free(ellValue);
  free(h_ellValues);
  return EXIT_SUCCESS;
}


