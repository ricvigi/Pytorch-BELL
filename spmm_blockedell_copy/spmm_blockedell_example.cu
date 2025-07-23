#include "myHeaders.cuh"


int
main(int argc, char** argv)
{
  std::string dtype = argv[5];
  if (dtype == "float")
  {
    return run<float>(argc, argv);
  } else if (dtype == "double")
  {
    return run<double>(argc, argv);
  } else
  {
    std::cerr << "Unsupported dtype: " << dtype << std::endl;
    return EXIT_UNSUPPORTED;
  }
}


template <typename T>
int run (int argc, char **argv)
{
  // Host problem definition
  unsigned int A_rows = atoi(argv[1]);
  unsigned int A_cols = atoi(argv[2]);
  float threshold = atof(argv[3]);
  int PRINT_DEBUG = atoi(argv[4]);

  torch::ScalarType dtype = torch::CppTypeToScalarType<T>::value;
  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  unsigned int B_rows = A_cols;
  unsigned int B_cols = A_cols;
  unsigned int C_rows = A_rows;
  unsigned int C_cols = B_cols;
  unsigned int lda = A_cols;
  unsigned int ldb = B_cols;
  unsigned int ldc = C_cols;

  torch::Tensor A = torch::randn({A_rows, A_cols}, torch::dtype(dtype));
  torch::Tensor B = torch::randn({B_rows, B_cols}, torch::dtype(dtype));
  A.masked_fill_(A < threshold, 0);
  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::dtype(dtype));

  T alpha = T(1);
  T beta  = T(0);


  T *hA = A.contiguous().data_ptr<T>();
  T *hB = B.contiguous().data_ptr<T>();
  T *hC = C.contiguous().data_ptr<T>();

  // count how many non zero values A has
  unsigned int n_non_zeroes = 0;
  count_non_zeroes<T>(hA, A_rows, A_cols, &n_non_zeroes);
  printf("number of non zeroes in A: %d\n", n_non_zeroes);

  // put the non zero values of A into a contiguous array
  T *non_zero_values = (T*) malloc(n_non_zeroes*sizeof(T));
  extract_non_zeros<T>(hA, A_rows, A_cols, non_zero_values);

  // Get the ellColInd array for matrix A
  int ellBlockSize, ellCols;
  int* ellColInd = nullptr;
  T* ellValue = nullptr;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  T *dB, *dC;
  CHECK_CUDA(cudaMallocAsync((void**) &dB, B_rows * B_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dC, C_rows * C_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMemcpyAsync(dB, hB, B_rows * B_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dC, hC, C_rows * C_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))

  /* [BEGIN] Dense to sparse conversion */

  // To create a conversion you need a dense matrix to convert it into a sparse matrix. If you want to store matrix A
  // in a sparse format, you need to convert A's dense representation to sparse!
  cusparseDnMatDescr_t matA = nullptr;
  cusparseSpMatDescr_t matSpA = nullptr;
  int *dA_columns = nullptr;
  T *dA_values = nullptr;
  T *dA_dense = nullptr;

  convert_to_blockedell<T>(A, matA, matSpA, dA_columns, dA_values, dA_dense, &ellBlockSize, &ellCols, ellColInd, ellValue);

  /* [END] Dense to sparse conversion */


  /* [BEGIN] Execute sparse-dense matrix multiplication */

  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, dC,
                                      cuda_type, CUSPARSE_ORDER_ROW) )

  execute_spmm<T>(matSpA, matB, matC, alpha, beta);


  /* [END] Execute sparse-dense matrix multiplication */

  if (PRINT_DEBUG > 0)
  {
    CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMM result:\n");
    for (unsigned int i = 0; i < C_rows; ++i)
    {
      for (unsigned int j = 0; j < C_cols; ++j)
      {
        printf("%f ", hC[i * C_cols + j]);
      }
      printf("\n");
    }

    torch::Tensor res = torch::mm(A,B).to(dtype);
    printf("PyTorch result:\n");
    std::cout << res << std::endl;
  }


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(dA_dense))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matSpA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  free(non_zero_values);
  free(ellColInd);
  free(ellValue);
  return EXIT_SUCCESS;
}
