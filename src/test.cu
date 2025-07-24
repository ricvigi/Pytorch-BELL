#include "myHeaders.cuh"

template <typename T>
__host__ int run(int argc, char **argv)
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
  unsigned int ldb = B_cols;
  unsigned int ldc = C_cols;

  torch::Tensor A = torch::randn({A_rows, A_cols}, torch::dtype(dtype));
  torch::Tensor B = torch::randn({B_rows, B_cols}, torch::dtype(dtype));
  torch::Tensor vector_X = torch::randn({A_cols}, torch::dtype(dtype));
  A.masked_fill_(A < T(threshold), T(0));
  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::dtype(dtype));
  torch::Tensor vector_Y = torch::zeros({A_rows}, torch::dtype(dtype));

  T alpha = T(1);
  T beta  = T(0);


  T *hA = A.contiguous().data_ptr<T>();
  T *hB = B.contiguous().data_ptr<T>();
  T *hC = C.contiguous().data_ptr<T>();
  T *h_vector_X = vector_X.contiguous().data_ptr<T>();
  T *h_vector_Y = vector_Y.contiguous().data_ptr<T>();

  // count how many non zero values A has
  unsigned int n_non_zeroes = 0;
  count_non_zeros<T>(hA, A_rows, A_cols, &n_non_zeroes);
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
  T *d_vector_X, *d_vector_Y;
  CHECK_CUDA(cudaMallocAsync((void**) &dB, B_rows * B_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dC, C_rows * C_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_X, A_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_Y, A_rows * sizeof(T), stream))
  CHECK_CUDA(cudaMemcpyAsync(dB, hB, B_rows * B_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dC, hC, C_rows * C_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_X, h_vector_X, A_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_Y, h_vector_Y, A_rows * sizeof(T), cudaMemcpyHostToDevice, stream))
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
  cusparseDnMatDescr_t vecX, vecY; // NOTE: Not actually vectors, but matrices with one dimension equal to 1. cusparse does not support spmv with blockedell.
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, dC,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  // Create dense vector X
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecX, A_cols, 1, 1, d_vector_X, cuda_type, CUSPARSE_ORDER_ROW) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecY, A_rows, 1, 1, d_vector_Y, cuda_type, CUSPARSE_ORDER_ROW) )

  execute_spmm<T>(matSpA, matB, matC, alpha, beta);
  execute_spmv<T>(matSpA, vecX, vecY, alpha, beta);


  /* [END] Execute sparse-dense matrix multiplication */

  if (PRINT_DEBUG > 0)
  {
    CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMM result:\n");
    for (unsigned int i = 0; i < C_rows; ++i)
    {
      for (unsigned int j = 0; j < C_cols; ++j)
      {
        std::cout << hC[i * C_cols + j] << " ";
      }
      printf("\n");
    }
    CHECK_CUDA(cudaMemcpy(h_vector_Y, d_vector_Y, A_rows * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMV result:\n");
    for (unsigned int i = 0; i < A_rows; ++i)
    {
      std::cout << h_vector_Y[i] << " ";
    }
    printf("\n");


    torch::Tensor res = torch::mm(A,B).to(dtype);
    torch::Tensor spmv_res = torch::mv(A,vector_X).to(dtype);
    printf("PyTorch result:\n");
    std::cout << res << std::endl;
    printf("PyTorch SpMV result:\n");
    std::cout << spmv_res << std::endl;
  }


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(d_vector_X))
  CHECK_CUDA(cudaFree(d_vector_Y))
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


template <typename T>
__host__ int run_int(int argc, char **argv)
{
 // Host problem definition
  unsigned int A_rows = atoi(argv[1]);
  unsigned int A_cols = atoi(argv[2]);
  unsigned int threshold = atoi(argv[3]);
  int PRINT_DEBUG = atoi(argv[4]);

  torch::ScalarType dtype = torch::CppTypeToScalarType<T>::value;
  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  unsigned int B_rows = A_cols;
  unsigned int B_cols = A_cols;
  unsigned int C_rows = A_rows;
  unsigned int C_cols = B_cols;
  unsigned int ldb = B_cols;
  unsigned int ldc = C_cols;

  torch::Tensor A = torch::randint(-128, 127, {A_rows, A_cols}, torch::dtype(dtype));
  torch::Tensor B = torch::randint(-128, 127, {B_rows, B_cols}, torch::dtype(dtype));
  torch::Tensor vector_X = torch::randint(-128, 127, {A_cols}, torch::dtype(dtype));
  A.masked_fill_(A < T(threshold), T(0));
  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::dtype(dtype));
  torch::Tensor vector_Y = torch::zeros({A_rows}, torch::dtype(dtype));

  T alpha = T(1);
  T beta  = T(0);


  T *hA = A.contiguous().data_ptr<T>();
  T *hB = B.contiguous().data_ptr<T>();
  T *hC = C.contiguous().data_ptr<T>();
  T *h_vector_X = vector_X.contiguous().data_ptr<T>();
  T *h_vector_Y = vector_Y.contiguous().data_ptr<T>();



  // count how many non zero values A has
  unsigned int n_non_zeroes = 0;
  count_non_zeros<T>(hA, A_rows, A_cols, &n_non_zeroes);
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
  T *d_vector_X, *d_vector_Y;
  CHECK_CUDA(cudaMallocAsync((void**) &dB, B_rows * B_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dC, C_rows * C_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_X, A_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_Y, A_rows * sizeof(T), stream))
  CHECK_CUDA(cudaMemcpyAsync(dB, hB, B_rows * B_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dC, hC, C_rows * C_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_X, h_vector_X, A_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_Y, h_vector_Y, A_rows * sizeof(T), cudaMemcpyHostToDevice, stream))
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
  cusparseDnMatDescr_t vecX, vecY; // NOTE: Not actually vectors, but matrices with one dimension equal to 1. cusparse does not support spmv with blockedell.
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, dC,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  // Create dense vector X
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecX, A_cols, 1, 1, d_vector_X, cuda_type, CUSPARSE_ORDER_ROW) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecY, A_rows, 1, 1, d_vector_Y, cuda_type, CUSPARSE_ORDER_ROW) )

  execute_spmm<T>(matSpA, matB, matC, alpha, beta);
  execute_spmv<T>(matSpA, vecX, vecY, alpha, beta);


  /* [END] Execute sparse-dense matrix multiplication */

  if (PRINT_DEBUG > 0)
  {
    CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMM result:\n");
    for (unsigned int i = 0; i < C_rows; ++i)
    {
      for (unsigned int j = 0; j < C_cols; ++j)
      {
        std::cout << hC[i * C_cols + j] << " ";
      }
      printf("\n");
    }
    CHECK_CUDA(cudaMemcpy(h_vector_Y, d_vector_Y, A_rows * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMV result:\n");
    for (unsigned int i = 0; i < A_rows; ++i)
    {
      std::cout << h_vector_Y[i] << " ";
    }
    printf("\n");


    torch::Tensor res = torch::mm(A,B).to(dtype);
    torch::Tensor spmv_res = torch::mv(A,vector_X).to(dtype);
    printf("PyTorch result:\n");
    std::cout << res << std::endl;
    printf("PyTorch SpMV result:\n");
    std::cout << spmv_res << std::endl;
  }


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(d_vector_X))
  CHECK_CUDA(cudaFree(d_vector_Y))
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

template <>
__host__ int run_int<int8_t>(int argc, char **argv)
{
  // ----------------------------- ATTENTION -----------------------------
  // if A and B are int8_t, to use spmm C must be of type CUDA_R_32I, i.e. int

  using T = int8_t;
  // Host problem definition
  int8_t A_rows = atoi(argv[1]);
  int8_t A_cols = atoi(argv[2]);
  int8_t threshold = atoi(argv[3]);
  int PRINT_DEBUG = atoi(argv[4]);

  torch::ScalarType dtype = torch::CppTypeToScalarType<T>::value;
  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  int8_t B_rows = A_cols;
  int8_t B_cols = A_cols;
  int8_t C_rows = A_rows;
  int8_t C_cols = B_cols;
  int8_t ldb = B_cols;
  int8_t ldc = C_cols;

  torch::Tensor A = torch::randint(-128, 127, {A_rows, A_cols}, torch::dtype(torch::kInt32)).to(torch::dtype(dtype));
  torch::Tensor B = torch::randint(-128, 127, {B_rows, B_cols}, torch::dtype(torch::kInt32)).to(torch::dtype(dtype));
  torch::Tensor vector_X = torch::randint(-128, 127, {A_cols}, torch::dtype(torch::kInt32)).to(torch::dtype(dtype));
  A.masked_fill_(A < T(threshold), T(0));
  std::cout << A << std::endl;
  torch::Tensor C = torch::zeros({A_rows, B_cols}, torch::dtype(torch::kInt32));
  std::cout << C << std::endl;
  torch::Tensor vector_Y = torch::zeros({A_rows}, torch::dtype(torch::kInt32));
  std::cout << vector_Y << std::endl;

  T alpha = T(1);
  T beta  = T(0);


  T *hA = A.contiguous().data_ptr<T>();
  T *hB = B.contiguous().data_ptr<T>();
  int *hC = C.contiguous().data_ptr<int>();
  T *h_vector_X = vector_X.contiguous().data_ptr<T>();
  int *h_vector_Y = vector_Y.contiguous().data_ptr<int>();

  // count how many non zero values A has
  // unsigned int n_non_zeroes = 0;
  // count_non_zeros<T>(hA, A_rows, A_cols, &n_non_zeroes);
  // printf("number of non zeroes in A: %d\n", n_non_zeroes);

  // put the non zero values of A into a contiguous array
  // T *non_zero_values = (T*) malloc(n_non_zeroes*sizeof(T));
  // extract_non_zeros<T>(hA, A_rows, A_cols, non_zero_values);

  // Get the ellColInd array for matrix A
  int ellBlockSize, ellCols;
  int* ellColInd = nullptr;
  T* ellValue = nullptr;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))
  T *dB;
  int *dC;
  T *d_vector_X;
  int *d_vector_Y;
  CHECK_CUDA(cudaMallocAsync((void**) &dB, B_rows * B_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dC, C_rows * C_cols * sizeof(int), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_X, A_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &d_vector_Y, A_rows * sizeof(int), stream))
  CHECK_CUDA(cudaMemcpyAsync(dB, hB, B_rows * B_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dC, hC, C_rows * C_cols * sizeof(int), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_X, h_vector_X, A_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_vector_Y, h_vector_Y, A_rows * sizeof(int), cudaMemcpyHostToDevice, stream))
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
  cusparseDnMatDescr_t vecX, vecY; // NOTE: Not actually vectors, but matrices with one dimension equal to 1. cusparse does not support spmv with blockedell.
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                      cuda_type, CUSPARSE_ORDER_ROW) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, dC,
                                      CUDA_R_32I, CUSPARSE_ORDER_ROW) )
  // Create dense vector X
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecX, A_cols, 1, 1, d_vector_X, cuda_type, CUSPARSE_ORDER_ROW) )
  // Create dense vector y
  CHECK_CUSPARSE( cusparseCreateDnMat(&vecY, A_rows, 1, 1, d_vector_Y, CUDA_R_32I, CUSPARSE_ORDER_ROW) )

  execute_spmm<T>(matSpA, matB, matC, alpha, beta);
  execute_spmv<T>(matSpA, vecX, vecY, alpha, beta);


  /* [END] Execute sparse-dense matrix multiplication */

  if (PRINT_DEBUG > 0)
  {
    CHECK_CUDA(cudaMemcpy(hC, dC, C_rows * C_cols * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMM result:\n");
    for (unsigned int i = 0; i < C_rows; ++i)
    {
      for (unsigned int j = 0; j < C_cols; ++j)
      {
        std::cout << hC[i * C_cols + j] << " ";
      }
      printf("\n");
    }
    CHECK_CUDA(cudaMemcpy(h_vector_Y, d_vector_Y, A_rows * sizeof(T), cudaMemcpyDeviceToHost))
    printf("SpMV result:\n");
    for (unsigned int i = 0; i < A_rows; ++i)
    {
      std::cout << h_vector_Y[i] << " ";
    }
    printf("\n");


    torch::Tensor res = torch::mm(A,B).to(dtype);
    torch::Tensor spmv_res = torch::mv(A,vector_X).to(dtype);
    printf("PyTorch result:\n");
    std::cout << res << std::endl;
    printf("PyTorch SpMV result:\n");
    std::cout << spmv_res << std::endl;
  }


  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC))
  CHECK_CUDA(cudaFree(d_vector_X))
  CHECK_CUDA(cudaFree(d_vector_Y))
  CHECK_CUDA(cudaFree(dA_dense))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matSpA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  // free(non_zero_values);
  free(ellColInd);
  free(ellValue);
  return EXIT_SUCCESS;
}


template __host__ int run<float>(int argc, char **argv);
template __host__ int run<double>(int argc, char **argv);

template __host__ int run_int<int8_t>(int argc, char **argv);
template __host__ int run_int<int>(int argc, char **argv);


