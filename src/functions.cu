#include "myHeaders.cuh"



__host__ int computeZeroBlocks (torch::Tensor &A, int rows, int cols, int kernelSize)
{
  int nBlocksH, nBlocksW, res;
  torch::Tensor B, bSums;
  nBlocksH = rows / kernelSize;
  nBlocksW = cols / kernelSize;
  B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
  B = B.permute({0, 2, 1, 3});
  bSums = B.sum({2, 3});
  res = (bSums == 0).sum().item<int>();
  return res;
}

/**
 * @brief Computes the number of zero blocks of size kernelSize in tensor A with an iterative approach, i.e. without calling pytorch's API. It's roughly four times quicker than its API counterpart.
 *
 * @param &A Reference to the tensor
 * @param rows Size of the first dimension of A
 * @param cols Size of the second dimension of A
 * @param kernelSize Size of the blocks (square)
 *
 * @return int, the number of zero blocks of size kernelSize x kernelSize in tensor A
 */
template <typename T>
__host__ int iterativeComputeZeroBlocks (torch::Tensor &A, int rows, int cols, int kernelSize)
{
  int count = 0;
  int nBlocksH = rows / kernelSize;
  int nBlocksW = cols / kernelSize;
  std::vector<T*> rowPointers (rows);
  # pragma omp parallel
  {
    #   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<T>();
    }
    #   pragma omp for collapse(2) reduction(+:count)
    for (int i = 0; i < nBlocksH; ++i)
    {
      for (int j = 0; j < nBlocksW; ++j)
      {
        bool isZeroBlock = true;
        for (int ii = 0; ii < kernelSize && isZeroBlock; ++ii)
        {
          for (int jj = 0; jj < kernelSize; ++jj)
          {
            int row = i * kernelSize + ii;
            int col = j * kernelSize + jj;
            if (rowPointers[row][col] != T(0))
            {
              isZeroBlock = false;
              break;
            }
          }
        }
        if (isZeroBlock) count++;
      }
    }
  }
  return count;
}

/**
 * @brief Computes the number of zero blocks of size kernelSize in tensor A
 *
 * @param &A Reference to the tensor
 * @param rows Size of the first dimension of A
 * @param cols Size of the second dimension of A
 * @param kernelSize Size of the blocks (square)
 *
 * @return returns a torch::Tensor object, that stores the sum of the values of all blocks of size kernelSize x kernelSize in A. From this object we will compute ellCols, but since we need it elsewhere, we return this object instead
 */
__host__ torch::Tensor computeEllCols (torch::Tensor& A, int rows, int cols, int kernelSize)
{
  int nBlocksH, nBlocksW;
  torch::Tensor B, bSums;
  nBlocksH = rows / kernelSize;
  nBlocksW = cols / kernelSize;
  B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
  B = B.permute({0, 2, 1, 3});, torch::dtype(torch::kFloat64));
  return bSums;
}

/**
 * @brief Iterative version of computeEllCols. Computes the number of zero blocks of size kernelSize in tensor A
 *
 * @param &A Reference to the tensor
 * @param rows Size of the first dimension of A
 * @param cols Size of the second dimension of A
 * @param kernelSize Size of the blocks (square)
 *
 * @return returns a torch::Tensor object, that stores the sum of the values of all blocks of size kernelSize x kernelSize in A. From this object we will compute ellCols, but since we need it elsewhere, we return this object instead
 */
template <typename T>
__host__ torch::Tensor iterativeComputeEllCols (torch::Tensor& A, int rows, int cols, int kernelSize)
{
  int nBlocksH = rows / kernelSize;
  int nBlocksW = cols / kernelSize;
  torch::Tensor bSums;
  T* tBSums;
  tBSums = (T*) malloc(rows*cols*sizeof(T));
  std::vector<T*> rowPointers (rows);
  auto del = [](void* ptr) { free(ptr); };

  # pragma omp parallel
  {
    #   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<T>();
    }
    #   pragma omp for collapse(2)
    for (int i = 0; i < nBlocksH; ++i)
    {
      for (int j = 0; j < nBlocksW; ++j)
      {
        T sum = T(0);
        for (int ii = 0; ii < kernelSize; ++ii)
        {
          for (int jj = 0; jj < kernelSize; ++jj)
          {
            int row = i * kernelSize + ii;
            int col = j * kernelSize + jj;
            T val = rowPointers[row][col];
            sum += val;
          }
        }
        tBSums[i * kernelSize + j] = sum;
      }
    }
  }
  // bSums = torch::from_blob(tBSums, {rows, cols}, torch::kFloat32).clone();
  // free(tBSums);
  bSums = torch::from_blob(tBSums, {rows, cols}, del, torch::TensorOptions().dtype(scalar_type<T>::val));
  return bSums;
}

/**
 * @brief Computes ellColInd array, i.e. the array that stores the index of the column position of all non-zero elements in A. If a row has < ellCols non-zero elements, the remaining elements in the rows will be set to -1
 *
 * @param &bSums Tensor that contains the sum of all elements in each ellBlockSize size of the original tensor
 * @param *ellColInd Pointer to the array that will store the index of the column position of all non-zero elements in A. This is our return value
 * @param rows Size of the first dimension of bSums
 * @param cols Size of the second dimension of bSums
 *
 * @return void. The return value of this function is ellColInd
 */
// template <typename T>
__host__ void getEllColInd (torch::Tensor &bSums, int *ellColInd, int rows, int cols)
{
  using T = double;
  int idx, rowSize;
  T val;

  std::vector<T*> rowPointers(rows);
  # pragma omp parallel shared(ellColInd) private(idx, rowSize, val)
  {
    #   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = bSums[i].contiguous().data_ptr<T>();
    }

    #   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      idx = 0;
      rowSize = 0;
      T* row = (T*) malloc(cols*sizeof(T));
      T* bSumsRow = rowPointers[i];
      for (int j = 0; j < bSums.size(1); ++j)
      {
        val = bSumsRow[j];
        if (val != T(0))
        {
          rowSize += 1;
          row[idx] = j;
          idx++;
        }
      }
      for (int j = 0; j < cols; ++j)
      {
        if (j < rowSize)
        {
          ellColInd[i*cols + j] = row[j];
        } else
        {
          ellColInd[i*cols + j] = T(-1);
        }
      }
      free(row);
    }
  }
}

/**
 * @brief Computes the blocked ellpack values array
 *
 * @param &A Reference to the tensor we have to transform
 * @param *ellValue Pointer to the array that will contain the blocked ell representation
 * @param *ellColInd Pointer to the indices array
 * @param rows Size of the first dimension of A, divided by ellBlockSize
 * @param cols Size of the second dimension of A, divided by ellBlockSize
 * @param ellBlockSize size of the blocks
 *
 * @return void. The return value of this function is ellValue
 */
template <typename T>
__host__ void getEllValues (torch::Tensor& A, T *ellValue, int *ellColInd, int rows, int cols, int ellBlockSize)
{
  int blockCol;
  int rowIndex;
  std::vector<T*> rowPointers(rows * ellBlockSize);

  # pragma omp parallel shared(rowPointers, A, ellColInd, ellValue) private(blockCol, rowIndex)
  {
    #   pragma omp for
    for (int i = 0; i < rows * ellBlockSize; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<T>();
    }

    #   pragma omp for collapse(2)
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        blockCol = ellColInd[i * cols + j];
        for (int bi = 0; bi < ellBlockSize; ++bi)
        {
          rowIndex = i * ellBlockSize + bi;
          T* rowA = rowPointers[rowIndex];
          int srcOffset = ellBlockSize * blockCol;
          int dstOffset = ((i * cols + j) * ellBlockSize + bi) * ellBlockSize;
          if (blockCol != -1)
          {
            memcpy(&ellValue[dstOffset], &rowA[srcOffset], ellBlockSize * sizeof(T));
          } else
          {
            std::fill(&ellValue[dstOffset], &ellValue[dstOffset + ellBlockSize], T(0));
          }
        }
      }
    }
  }
}

/**
 * @brief Find the block size that filters out the most zeroes
 *
 * @param &A Reference to sparse tensor that we want to convert to BELL
 * @param x Size of the first dimension of A
 * @param y Size of the second dimension of A
 * @param &ellBlockSize Reference to the variable that will store the best block size for tensor A
 * @param &ellCols Reference to the variable that will store the number of columns in BELL
 * @param *&ellColInd Pointer to a reference of the ellColInd array, that stores the indices of non zero blocks
 * @param *&ellValue Pointer to a reference to the ellValue array, that contains the non zero values of A
 *
 * @return int EXIT_SUCCESS or int EXIT_FAILURE
 */
template <typename T>
__host__ int getBellParams(torch::Tensor& A, int x, int y, int& ellBlockSize, int& ellCols, int*& ellColInd, T*& ellValue)
{
  /* Variable declarations */
  int i;
  int zeroCount;
  int *divisors;
  int divisorsSize;
  int rows;
  int cols;
  int size;
  int nThreads;

  double start, end, tStart, tEnd;
  torch::Tensor bSums;

  /*
   * ATTENTION: This instruction allows nested parallelism. Right now it improves performance, but i'm
   * not sure why :/...
   * If you remove this, or set it to 0, all calls to functions that contain parallel regions, if called
   * when already inside a parallel region, will be executed sequentially, not in parallel.
   */
  omp_set_nested(1);

  /* Matrix sizes checks */
  if (x != y) // TODO: Change this, matrix can also not be square, although the block must be square. This implies that both x and y must be divisibile by the chosen block-size.
  {
    printf("Matrix must be square\n");
    fflush(stdout);
    return EXIT_FAILURE;
  } else if (isPrime(x) == 1)
  {
    printf("Matrix dimensions can't be prime\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  /*
   *
   * START PROGRAM
   *
   */

  /* The optimal block size shape will be stored in ellBlockSize. */
  ellCols = 0;
  ellBlockSize = 0;
  zeroCount = 0;
  divisors = nullptr; /* Initialize the pointer */
  divisorsSize = findDivisors(x, divisors);
  nThreads = std::min(divisorsSize, omp_get_num_procs());
  omp_set_num_threads(nThreads);
  printf("divisorsSize: %d\n", divisorsSize);

  std::vector<int> localZeroCount(nThreads);
  std::vector<int> localKernels(nThreads);

  start = omp_get_wtime();
  tStart = omp_get_wtime();
  # pragma omp parallel shared(zeroCount, ellBlockSize, localZeroCount)
  {
    int localKernelSize;
    int localZeroBlocks;
    int localNZeroes;
    int localBestZeroes;
    int localBestKernel;
    int id;
    id = omp_get_thread_num();
    localBestZeroes = 0;

    /* NOTE: Should we use a guided schedule for this loop parallelism?
     *    /* ATTENTION: This routine might be a waste of time... remember that if blocksize 2*n has < zeroes than blocksize
     *    /* n, it might be pointless to continue... */
     #pragma omp single
     {
       std::cout << "1.1.1" << std::endl;
     }
     #   pragma omp for
     for (i = 0; i < divisorsSize; ++i)
     {

       localKernelSize = divisors[i];
       localZeroBlocks = iterativeComputeZeroBlocks<T>(A, x, y, localKernelSize);
       if (localZeroBlocks > 0)
       {
         localNZeroes = localZeroBlocks * localKernelSize * localKernelSize;
         if (localBestZeroes < localNZeroes)
         {
           localBestZeroes = localNZeroes;
           localBestKernel = localKernelSize;
         }
       } else
       {
         continue;
       }
     }
     localZeroCount[id] = localBestZeroes;
     localKernels[id] = localBestKernel;
  }
#pragma omp single
{
  std::cout << "1.1.2" << std::endl;
}
  /* Get the best block size value */
  int z, k;
  for (int i = 0; i < nThreads; ++i)
  {
    z = localZeroCount[i];
    k = localKernels[i];
    if (z > zeroCount)
    {
      zeroCount = z;
      ellBlockSize = k;
    } else
    {
      continue;
    }
  }
  tEnd = omp_get_wtime();

  printf("computeZeroBlocks time: %f\n", tEnd - tStart);

  omp_set_num_threads(omp_get_num_procs());

  tStart = omp_get_wtime();
  /* Now get ellCols, the actual number of columns in the BELL format */
  std::cout << "1.1.3" << std::endl;
  bSums = computeEllCols(A, x, y, ellBlockSize);
  std::cout << "1.1.35" << std::endl;
  // bSums = iterativeComputeEllCols<T>(A, x, y, ellBlockSize);
  ellCols = (bSums != 0).sum(1).max().item<int>();
  tEnd = omp_get_wtime();
  printf("computeEllCols time: %f\n", tEnd - tStart);

  /* Allocate memory for ellColInd array */
  rows = x / ellBlockSize;
  cols = ellCols;
  size = rows*cols;
  ellColInd = (int*) malloc(size*sizeof(int));

  tStart = omp_get_wtime();
  /* Create the ellColInd array */
  std::cout << "1.1.4" << std::endl;
  getEllColInd(bSums, ellColInd, rows, cols);
  std::cout << "1.1.45" << std::endl;
  tEnd = omp_get_wtime();
  printf("getEllColInd time: %f\n", tEnd - tStart);

  tStart = omp_get_wtime();
  /* Create the ellValue array */
  /* ATTENTION: How many rows does ellValue have? should we completely eliminate a row of blocks if it's entirely made of zeroes? */
  /* NOTE: OLD memory formula. If something breaks it might be due to the new one down below */
  // ellValue = (float*) malloc((rows*cols*ellBlockSize*ellBlockSize)*sizeof(float));
  /* NEW FORMULA */
  ellValue = (T*) malloc((x*ellCols*ellBlockSize)*sizeof(T));
  std::cout << "1.1.5" << std::endl;
  getEllValues<T>(A, ellValue, ellColInd, rows, cols, ellBlockSize);
  std::cout << "1.1.55" << std::endl;
  // getEllValues(A, ellValue, ellColInd);
  tEnd = omp_get_wtime();
  printf("getEllValues time: %f\n", tEnd - tStart);
  end = omp_get_wtime();
  /*
   *
   * END PROGRAM
   *
   */

  // if (PRINT_DEBUG)
  // {
  //   std::cout << A << std::endl;
  //   std::cout << bSums << std::endl;
  //   printMat(ellColInd, rows, cols);
  //   printEllValue(ellValue, rows, cols, ellBlockSize);
  // }

  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, ellBlockSize);
  printf("Matrix has %d zero blocks of size %d\n", zeroCount / (ellBlockSize*ellBlockSize), ellBlockSize);
  printf("Total time needed for computation: %7.6f\n", end - start);

  free(divisors);
  return EXIT_SUCCESS;
}

/**
 * @brief converts matrix A (pointer) into blockedell format and returns a descriptor object of the blockedell format
 */
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
                                   T *ellValue                 /* in */)
{
  unsigned int A_rows = A.size(0);
  unsigned int A_cols = A.size(1);
  std::cout << A_rows << " " << A_cols << std::endl;
  unsigned int lda = A_cols;

  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  T *hA = A.contiguous().data_ptr<T>();

  // Get the ellColInd array for matrix A
  int err;

  std::cout << "1.1" << std::endl;
  err = getBellParams<T>(A, A_rows, A_cols, *ellBlockSize, *ellCols, ellColInd, ellValue);
  std::cout << "1.2" << std::endl;
  if (err != 0)
  {
    std::cout << "Error code " << err << ",exiting!" << std::endl;
    fflush(stdout);
    return err;
  }

  // ATTENTION: ellCols is usually considered to be the number of columns in ell format, NOT the number of blocks (of the ell format).
  *ellCols = (*ellBlockSize) * (*ellCols);
  // Device memory management
  int ellColInd_size = A_rows * (*ellCols);
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  CHECK_CUDA(cudaMallocAsync((void**) &dA_dense, A_rows * A_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_columns, ellColInd_size * sizeof(int), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_values, A_rows * (*ellCols) * sizeof(T), stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_dense, hA, A_rows * A_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_columns, ellColInd, ellColInd_size * sizeof(int), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemsetAsync(dA_values, T(0), A_rows * (*ellCols) * sizeof(T), stream))
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
  CHECK_CUSPARSE(cusparseDestroy(conversionHandle))
  return EXIT_SUCCESS;
}

/* ATTENTION: This is a specialization for INT type, which requires the sparse matrix to have an 8 bit integer */
template <>
__host__ int convert_to_blockedell<int>(torch::Tensor &A            /* in */,
                                        cusparseDnMatDescr_t &matA  /* in */,
                                        cusparseSpMatDescr_t &spA   /* out */,
                                        int *dA_columns             /* in */,
                                        int *dA_values                /* in */,
                                        int *dA_dense                 /* in */,
                                        int *ellBlockSize           /* in */,
                                        int *ellCols                /* in */,
                                        int *ellColInd              /* in */,
                                        int *ellValue                 /* in */)
{
  using T = int;
  unsigned int A_rows = A.size(0);
  unsigned int A_cols = A.size(1);
  printf("%d %d\n", A_rows, A_cols);
  unsigned int lda = A_cols;

  constexpr cudaDataType_t cuda_type = CUDA_R_8I;

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
  int ellColInd_size = A_rows * (*ellCols);
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream))

  CHECK_CUDA(cudaMallocAsync((void**) &dA_dense, A_rows * A_cols * sizeof(T), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_columns, ellColInd_size * sizeof(int), stream))
  CHECK_CUDA(cudaMallocAsync((void**) &dA_values, A_rows * (*ellCols) * sizeof(T), stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_dense, hA, A_rows * A_cols * sizeof(T), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(dA_columns, ellColInd, ellColInd_size * sizeof(int), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemsetAsync(dA_values, T(0), A_rows * (*ellCols) * sizeof(T), stream))
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
  CHECK_CUSPARSE(cusparseDestroy(conversionHandle))
  return EXIT_SUCCESS;
}


/**
 * @brief This function execute sparse matrix-matrix multiplication and stores the result in dense matric C.
 * @note If you consider B as a m x 1 vector, it's also an implementation of sparse-vector multiplication
 */
template <typename T>
__host__ int execute_spmm(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, T alpha, T beta)
{
  cusparseHandle_t spmm_handle = NULL;
  void* d_spmm_handle_buffer = NULL;
  size_t d_spmm_handle_buffer_size = 0;
  CHECK_CUSPARSE( cusparseCreate(&spmm_handle) )

  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
    spmm_handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, spA, B, &beta, C, cuda_type,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &d_spmm_handle_buffer_size) )

  CHECK_CUDA( cudaMalloc(&d_spmm_handle_buffer, d_spmm_handle_buffer_size) )

  // execute SpMM
  CHECK_CUSPARSE( cusparseSpMM(spmm_handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, spA, B, &beta, C, cuda_type,
                               CUSPARSE_SPMM_BLOCKED_ELL_ALG1, d_spmm_handle_buffer) )
  CHECK_CUSPARSE( cusparseDestroy(spmm_handle) )
  CHECK_CUDA( cudaFree(d_spmm_handle_buffer) )
  return EXIT_SUCCESS;
}

/**
 * @brief This function executes sparse matrix-vector multiplication and stores the result in vecY.
 * @note Because cusparse does not support spmv for blockedell format, we have to trick the user and perform spmm instead of spmv, where the vector is a m x 1 matrix.
 */
template <typename T>
__host__ int execute_spmv(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t vecX, cusparseDnMatDescr_t vecY, T alpha, T beta)
{
  cusparseHandle_t spmv_handle = NULL;
  void *d_spmv_buffer = NULL;
  size_t d_spmv_buffer_size = 0;
  CHECK_CUSPARSE( cusparseCreate(&spmv_handle) )

  constexpr cudaDataType_t cuda_type = cuda_dtype<T>::val;

  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(spmv_handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, spA, vecX, &beta, vecY, cuda_type,
                                          CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &d_spmv_buffer_size))
  CHECK_CUDA( cudaMalloc(&d_spmv_buffer, d_spmv_buffer_size) )

  // execute SpMV
  CHECK_CUSPARSE( cusparseSpMM(spmv_handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, spA, vecX, &beta, vecY, cuda_type,
                               CUSPARSE_SPMM_BLOCKED_ELL_ALG1, d_spmv_buffer))
  CHECK_CUSPARSE(cusparseDestroy(spmv_handle))
  CHECK_CUDA(cudaFree(d_spmv_buffer))
  return EXIT_SUCCESS;
}



/* [BEGIN] Function instantiations */

template __host__ int iterativeComputeZeroBlocks<float>(torch::Tensor&, int, int, int);
template __host__ int iterativeComputeZeroBlocks<double>(torch::Tensor&, int, int, int);
template __host__ int iterativeComputeZeroBlocks<int8_t>(torch::Tensor&, int, int, int);
template __host__ int iterativeComputeZeroBlocks<int>(torch::Tensor&, int, int, int);

template __host__ torch::Tensor iterativeComputeEllCols<float>(torch::Tensor&, int, int, int);
template __host__ torch::Tensor iterativeComputeEllCols<double>(torch::Tensor&, int, int, int);
template __host__ torch::Tensor iterativeComputeEllCols<int8_t>(torch::Tensor&, int, int, int);
template __host__ torch::Tensor iterativeComputeEllCols<int>(torch::Tensor&, int, int, int);

// template __host__ void getEllColInd<float>(torch::Tensor&, int*, int, int);
// template __host__ void getEllColInd<double>(torch::Tensor&, int*, int, int);
// template __host__ void getEllColInd<int8_t>(torch::Tensor&, int*, int, int);
// template __host__ void getEllColInd<int>(torch::Tensor&, int*, int, int);

template __host__ void getEllValues<float>(torch::Tensor&, float*, int*, int, int, int);
template __host__ void getEllValues<double>(torch::Tensor&, double*, int*, int, int, int);
template __host__ void getEllValues<int8_t>(torch::Tensor&, int8_t*, int*, int, int, int);
template __host__ void getEllValues<int>(torch::Tensor&, int*, int*, int, int, int);

template __host__ int getBellParams<float>(torch::Tensor&, int, int, int&, int&, int*&, float*&);
template __host__ int getBellParams<double>(torch::Tensor&, int, int, int&, int&, int*&, double*&);
template __host__ int getBellParams<int8_t>(torch::Tensor&, int, int, int&, int&, int*&, int8_t*&);
template __host__ int getBellParams<int>(torch::Tensor&, int, int, int&, int&, int*&, int*&);

template __host__ int convert_to_blockedell<double>(torch::Tensor &A , cusparseDnMatDescr_t &matA, cusparseSpMatDescr_t &spA, int *dA_columns, double *dA_values,
                                                    double *dA_dense, int *ellBlockSize, int *ellCols, int *ellColInd, double *ellValue);
template __host__ int convert_to_blockedell<float>(torch::Tensor &A , cusparseDnMatDescr_t &matA, cusparseSpMatDescr_t &spA, int *dA_columns, float *dA_values,
                                                   float *dA_dense, int *ellBlockSize, int *ellCols, int *ellColInd, float *ellValue);
template __host__ int convert_to_blockedell<int8_t>(torch::Tensor &A , cusparseDnMatDescr_t &matA, cusparseSpMatDescr_t &spA, int *dA_columns, int8_t *dA_values,
                                                    int8_t *dA_dense, int *ellBlockSize, int *ellCols, int *ellColInd, int8_t *ellValue);
template __host__ int convert_to_blockedell<int>(torch::Tensor &A , cusparseDnMatDescr_t &matA, cusparseSpMatDescr_t &spA, int *dA_columns, int *dA_values,
                                                 int *dA_dense, int *ellBlockSize, int *ellCols, int *ellColInd, int *ellValue);

template __host__ int execute_spmm<double>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, double alpha, double beta);
template __host__ int execute_spmm<float>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, float alpha, float beta);
template __host__ int execute_spmm<int8_t>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, int8_t alpha, int8_t beta);
template __host__ int execute_spmm<int>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C, int alpha, int beta);

template __host__ int execute_spmv<double>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t vecX, cusparseDnMatDescr_t vecY, double alpha, double beta);
template __host__ int execute_spmv<float>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t vecX, cusparseDnMatDescr_t vecY, float alpha, float beta);
template __host__ int execute_spmv<int8_t>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t vecX, cusparseDnMatDescr_t vecY, int8_t alpha, int8_t beta);
template __host__ int execute_spmv<int>(cusparseSpMatDescr_t spA, cusparseDnMatDescr_t vecX, cusparseDnMatDescr_t vecY, int alpha, int beta);


/* [END] Function instantiations */

