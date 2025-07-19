#include "myHeaders.hpp"

// int PRINT_DEBUG = 0;

int
main(int argc, char** argv)
{
  // Host problem definition
  unsigned int A_rows = atoi(argv[1]);
  unsigned int A_cols = atoi(argv[2]);
  float threshold = atof(argv[3]);
  int PRINT_DEBUG = atoi(argv[4]);

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


  float *hA = A.contiguous().data_ptr<float>();
  float *hB = B.contiguous().data_ptr<float>();
  float *hC = C.contiguous().data_ptr<float>();

  // count how many non zero values A has
  unsigned int n_non_zeroes = 0;
  count_non_zeroes(hA, A_rows, A_cols, &n_non_zeroes);
  printf("number of non zeroes in A: %d\n", n_non_zeroes);

  // put the non zero values of A into a contiguous array
  float *non_zero_values = (float*) malloc(n_non_zeroes*sizeof(float));
  extract_non_zeros(hA, A_rows, A_cols, non_zero_values);

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
  // printf("ellCols: %d, n_non_zeroes: %d\n", ellCols, n_non_zeroes);
  int ellColInd_size = A_rows * ellCols;
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  int *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA(cudaMallocAsync((void**) &dA_columns, ellColInd_size * sizeof(int), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_values, A_rows * ellCols * sizeof(float), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dB, B_rows * B_cols * sizeof(float), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dC, C_rows * C_cols * sizeof(float), stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_columns, ellColInd, ellColInd_size * sizeof(int), cudaMemcpyHostToDevice, stream))
  // CHECK_CUDA(cudaMemcpy(dA_values, ellValue, A_rows * ellCols * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemsetAsync(dA_values, 0.0f, A_rows * ellCols * sizeof(float), stream))
  CHECK_CUDA(cudaMemcpyAsync(dB, hB, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dC, hC, C_rows * C_cols * sizeof(float), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))

  printf("ellCols: %d, ellBlockSize: %d\n", ellCols, ellBlockSize);

  /* [BEGIN] Dense to sparse conversion */
  // To create a conversion you need a dense matrix to convert it into a sparse matrix. If you want to store matrix A
  // in a sparse format, you need to convert A's dense representation to sparse!
  cusparseHandle_t     conversionHandle = NULL;
  cusparseDnMatDescr_t matA;
  cusparseSpMatDescr_t matSpA;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&conversionHandle))

  /* [BEGIN] Create events to time the runtime of spmm */
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start))
  CHECK_CUDA(cudaEventCreate(&stop))
  /* [END] Create events to time the runtime of spmm */



  /* ATTENTION: remember that leading dimension is number of columns if we use CUSPARSE_ORDER_ROW, and vice versa */
  // Create dense matrix A
  float *dA_dense;
  CHECK_CUDA(cudaMallocAsync((void**) &dA_dense, A_rows * A_cols * sizeof(double), stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_dense, hA, A_rows * A_cols * sizeof(double), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))

  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_rows, A_cols, lda, dA_dense,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )

  // Create sparse matrix B in Blocked ELL format
  CHECK_CUSPARSE( cusparseCreateBlockedEll(&matSpA, A_rows, A_cols,
                                           ellBlockSize, ellCols,
                                           dA_columns, dA_values,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_32F) )

  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(conversionHandle, matA, matSpA,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(conversionHandle, matA, matSpA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_convert(conversionHandle, matA, matSpA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))
  /* [END] Dense to sparse conversion */


  /* [BEGIN] Execute sparse-dense matrix multiplication */

  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )

  execute_spmm<float>(matSpA, matB, matC, alpha, beta);


  /* [END] Execute sparse-dense matrix multiplication */

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
  if (PRINT_DEBUG > 0)
  {
    CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost))
    printf("SpMM result:\n");
    for (unsigned int i = 0; i < C_rows; ++i)
    {
      for (unsigned int j = 0; j < C_cols; ++j)
      {
        printf("%f ", hC[i * C_cols + j]);
      }
      printf("\n");
    }

    torch::Tensor res = torch::mm(A,B);
    printf("PyTorch result:\n");
    std::cout << res << std::endl;
  }


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_dense))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matSpA))
  CHECK_CUSPARSE(cusparseDestroy(conversionHandle))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  CHECK_CUDA(cudaEventDestroy(start))
  CHECK_CUDA(cudaEventDestroy(stop))
  free(non_zero_values);
  free(ellColInd);
  free(ellValue);
  free(h_ellValues);
  return EXIT_SUCCESS;
}


