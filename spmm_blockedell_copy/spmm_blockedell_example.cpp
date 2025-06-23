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


int main(int argc, char** argv)
{
  PRINT_DEBUG = atoi(argv[4]);
  // Host problem definition
  int A_num_rows = atoi(argv[1]);
  int A_num_cols = atoi(argv[2]);
  float threshold = atof(argv[3]);
  torch::Tensor A = torch::randn({A_num_rows, A_num_cols});
  A.masked_fill_(A < threshold, 0);
  torch::Tensor E = torch::randn({A_num_cols, A_num_cols});

  torch::Tensor res = torch::mm(A, E);

  int A_ell_blocksize;
  int A_ell_cols;
  int *A_ellColInd;
  float  *ellValue;


  int   B_num_rows      = A_num_rows;
  int   B_num_cols      = A_num_cols;
  int   ldb             = B_num_cols;
  int   ldc             = A_num_cols;
  int   ldd             = A_num_cols;
  int   lde             = A_num_cols;
  int   A_size          = A_num_rows*A_num_cols;
  int   B_size          = A_num_rows*A_num_cols;
  int   C_size          = A_num_rows*A_num_cols;
  int   E_size          = A_num_cols*A_num_cols;
  float *hC = (float*)malloc(C_size * sizeof(float));
  float *hE = (float*)malloc(E_size * sizeof(float));
  float alpha           = 1.0f;
  float beta            = 0.0f;
  int err = getBellParams(A, A_num_rows, A_num_cols, A_ell_blocksize, A_ell_cols, A_ellColInd, ellValue);

  /* Turn A_ellColInd from row major to column major */
  // int *A_ellColInd_col_major = (int*) malloc((A_ell_cols / A_ell_blocksize) * (A_num_rows / A_ell_blocksize) * sizeof(int));
  // rowToColMajor(A_ellColInd, A_ellColInd_col_major, (A_num_rows / A_ell_blocksize), (A_ell_cols / A_ell_blocksize));
  if (err != 0)
  {
    std::printf("Error code %d, exiting!\n", err);
    fflush(stdout);
    return err;
  }

  int A_ell_rows = A_num_rows / A_ell_blocksize;
  //--------------------------------------------------------------------------
  // Check compute capability
  cudaDeviceProp props;
  CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
  if (props.major < 7)
  {
    std::printf("cusparseSpMM with blocked ELL format is supported only "
    "with compute capability at least 7.0\n");
    return EXIT_UNSUPPORTED;
  }
  //--------------------------------------------------------------------------
  // Device memory management
  float *h_dense;
  int   *d_ell_columns;
  float *d_ell_values,  *d_dense, *dC, *dD, *dE;

  h_dense = A.contiguous().data_ptr<float>();

  /* C is a temporary array that is used between the conversion of A into it's blocked-ell version, that in this
  /*  program is the variable B */
  float* C = (float*) calloc(A_num_rows * A_ell_cols * A_ell_blocksize, sizeof(float));
  float* D = (float*) calloc(A_num_rows * A_num_cols, sizeof(float));
  std::cout << "[START]: Sparse conversion memory allocation" << std::endl;

  CHECK_CUDA( cudaMalloc((void**) &d_dense, A_size * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &d_ell_columns,
                         A_ell_blocksize * A_ell_cols * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &d_ell_values,
                         A_num_rows * A_ell_cols * A_ell_blocksize *  sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC, A_num_rows * A_num_cols * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dD, A_num_rows * A_num_cols * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dE, A_num_cols * A_num_cols * sizeof(float)) )
  CHECK_CUDA( cudaMemcpy(d_dense, h_dense, A_size * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_ell_columns, A_ellColInd,
                         A_ell_blocksize * A_ell_cols * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_ell_values, C,
                         A_num_rows * A_ell_cols * A_ell_blocksize * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dE, E.contiguous().data_ptr<float>(),
                         A_num_cols * A_num_rows * sizeof(float),
                         cudaMemcpyHostToDevice) )
  /* ATTENTION: Size of dC is A_num_rows * E_num_cols. Now it works but you need to change this for the logic
  /* to be correct */
  CHECK_CUDA( cudaMemset(dC, 0.0f, A_num_rows * A_num_cols * sizeof(float)) )
  CHECK_CUDA( cudaMemset(dD, 0.0f, A_num_rows * A_num_cols * sizeof(float)) )

  std::cout << "[END-OK]: Sparse conversion memory allocation" << std::endl;
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseDnMatDescr_t matD;
  cusparseSpMatDescr_t matB; /* NOTE: Sparse and dense matrices have their respective descriptors */
  cusparseDnMatDescr_t matA;
  cusparseDnMatDescr_t matE;
  cusparseDnMatDescr_t matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  std::cout << "[START]: Sparse conversion and multiplication" << std::endl;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, A_num_cols, d_dense,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // CHECK_CUDA( cudaDeviceSynchronize() )

  CHECK_CUSPARSE( cusparseCreateBlockedEll(&matB, A_num_rows, A_num_cols,
                                           A_ell_blocksize, A_ell_cols * A_ell_blocksize,
                                           d_ell_columns, d_ell_values,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_32F) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
    handle, matA, matB,
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    &bufferSize) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  //
  // execute Dense to Sparse conversion
  CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                                 CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                 dBuffer) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  // execute Dense to Sparse conversion
  CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                                CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                dBuffer) )
  // CHECK_CUDA( cudaDeviceSynchronize() )

  std::cout << "[***]B:" << std::endl;

  /* ATTENTION: Change CUSPARSE_ORDER_COL to CUSPARSE_ORDER_ROW */
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, A_num_cols, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matD, A_num_rows, A_num_cols, ldd, dD,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matE, A_num_cols, A_num_cols, lde, dE,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // CHECK_CUDA( cudaDeviceSynchronize() )

  void*                dBuffera    = NULL;
  size_t               bufferSizea = 0;
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matB, matE, &beta, matC, CUDA_R_32F,
                                          CUSPARSE_SPMM_ALG_DEFAULT, &bufferSizea) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  CHECK_CUDA( cudaMalloc(&dBuffera, bufferSizea) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matB, matE, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffera) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  std::cout << "[END-OK]: Sparse conversion and multiplication" << std::endl;

  std::cout << "[START]: Check results" << std::endl;
  CHECK_CUDA( cudaMemcpy(hE, dC, C_size * sizeof(float),
                         cudaMemcpyDeviceToHost) )
  // CHECK_CUDA( cudaDeviceSynchronize() )
  float *result = res.contiguous().data_ptr<float>();
  std::cout << "Device result:" << std::endl;
  printMat(hE, A_num_rows, A_num_cols);
  std::cout << "Host result:" << std::endl;
  printMat(result, A_num_rows, A_num_cols);
  float epsilon = 1e-6f;
  int correct = 1;
  for (int i = 0; i < A_num_rows; ++i)
  {
    for (int j = 0; j < B_num_cols; ++j)
    {
      float e_value = hE[i * A_num_cols + j];
      float e_result = result[i * A_num_cols + j];
      float relative_error = std::fabs(e_value - e_result) / (std::fabs(e_result) + epsilon);
      if (relative_error > 1e-3f)
      {
        correct = 0; // direct floating point comparison is not reliable
        std::printf("e_value: %f \ne_result: %f\n", e_value, e_result);
        break;
      }
    }
    if (!correct)
    {
      break;
    }
  }

  if (correct)
    std::printf("spmm_blockedell_example test PASSED\n");
  else
    std::printf("spmm_blockedell_example test FAILED: wrong result\n");

  std::cout << "[END-OK]: Check results" << std::endl;


  // destroy matrix/vector descriptors and deallocate memory
  std::cout << "[START]: Memory deallocation" << std::endl;
  CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
  CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  free(C);
  free(D);
  free(hC);
  free(hE);
  CHECK_CUDA( cudaFree(d_dense) )
  CHECK_CUDA( cudaFree(d_ell_columns) )
  CHECK_CUDA( cudaFree(d_ell_values) )
  CHECK_CUDA( cudaFree(dC) )
  CHECK_CUDA( cudaFree(dD) )
  CHECK_CUDA( cudaFree(dE) )
  CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(dBuffera) )
  std::cout << "[END-OK]: Memory deallocation" << std::endl;
  return EXIT_SUCCESS;
}


