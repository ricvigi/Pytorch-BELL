#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

int computeZeroBlocks(torch::Tensor& A , int rows, int cols, int kernelSize)
{
  /* Computes the number of zero blocks of size kernelSize in matrix A */
    int nBlocksH, nBlocksW;
    torch::Tensor B, bSums;
    nBlocksH = rows / kernelSize;
    nBlocksW = cols / kernelSize;
    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2, 3});
    return (bSums == 0).sum().item<int>();
}

torch::Tensor computeEllCols(torch::Tensor& A, int rows, int cols, int kernelSize)
{
  /*
   * This lambda returns a tensor that contains the sum of all blocks of size kernelSize in the matrix. ATTENTION: Do
   * NOT change the return value of this lambda again... it returns a tensor just get over with it
   */
  int nBlocksH, nBlocksW;
  torch::Tensor B, bSums;
  nBlocksH = rows / kernelSize;
  nBlocksW = cols / kernelSize;
  B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
  B = B.permute({0, 2, 1, 3});
  bSums = B.sum({2,3});
  return bSums;
}

void getEllColInd(torch::Tensor& bSums, int* ellColInd, int rows, int cols)
{
  /*
   * Compute the size of ellColInd array, i.e. the array that stoellBlockSize the index of the column position of all non-zero
   * elements in A. If a row has < ellCols non-zero elements, the remaining elements will be set to -1
   */
  int idx, rowSize;
  float val;

  std::vector<float*> rowPointers(rows);
# pragma omp parallel for
  for (int i = 0; i < rows; ++i)
  {
    rowPointers[i] = bSums[i].contiguous().data_ptr<float>();
  }

# pragma omp parallel for shared(ellColInd) private(idx, rowSize, val)
  for (int i = 0; i < rows; ++i)
  {
    idx = 0;
    rowSize = 0;
    int* row = (int*) malloc(cols*sizeof(int));
    float* bSumsRow = rowPointers[i];
    for (int j = 0; j < bSums.size(1); ++j)
    {
      val = bSumsRow[j];
      if (val != 0)
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
        ellColInd[i*cols + j] = -1;
      }
    }
    free(row);
  }
}

void getEllValues(torch::Tensor& A, float *ellValue, int *ellColInd, int rows, int cols, int ellBlockSize)
{
  /* This lambda gives the blocked ellpack values array */
  int blockCol, dstIndex, rowIndex, colIndex;
  std::vector<float*> rowPointers(rows * ellBlockSize);

# pragma omp parallel for
  for (int i = 0; i < rows * ellBlockSize; ++i)
  {
    rowPointers[i] = A[i].contiguous().data_ptr<float>();
  }

# pragma omp parallel for collapse(2) private(blockCol, dstIndex, rowIndex, colIndex)
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      blockCol = ellColInd[i * cols + j];
      for (int bi = 0; bi < ellBlockSize; ++bi)
      {
        rowIndex = i * ellBlockSize + bi;
        float* rowA = rowPointers[rowIndex];
        int srcOffset = ellBlockSize * blockCol;
        int dstOffset = ((i * cols + j) * ellBlockSize + bi) * ellBlockSize;
        if (blockCol != -1)
        {
          memcpy(&ellValue[dstOffset], &rowA[srcOffset], ellBlockSize * sizeof(float));
        } else
        {
          std::fill(&ellValue[dstOffset], &ellValue[dstOffset + ellBlockSize], 0.0f);
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
 *
 * @return Best block size on success, EXIT_FAILURE on error.
 */
int getBellParams(torch::Tensor& A, int x, int y, int& ellBlockSize, int& ellCols, int*& ellColInd, float*& ellValue)
{
  /* Variable declarations */
  int i, j, kernelSize, zeroBlocks, maxZeroBlocks, zeroCount, nZeroes, *divisors, divisorsSize, rows, cols, size, colIdx;
  float start, end;
  torch::Tensor bSums, block;

  /* Matrix sizes checks */
  if (x != y)
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
   * START LAMBDA DEFINITIONS
   *
   */


//   auto getEllValues = [&](torch::Tensor& A, float *ellValue, int *ellColInd) -> void
//   {
//     /* This lambda gives the blocked ellpack values array */
//     int blockCol, dstIndex, rowIndex, colIndex;
//
//     std::vector<float*> rowPointers(rows * ellBlockSize);
//
// #   pragma omp parallel for
//     for (int i = 0; i < rows * ellBlockSize; ++i)
//     {
//       rowPointers[i] = A[i].contiguous().data_ptr<float>();
//     }
//
// #   pragma omp parallel for collapse(2) private(blockCol, dstIndex, rowIndex, colIndex)
//     for (int i = 0; i < rows; ++i)
//     {
//       for (int j = 0; j < cols; ++j)
//       {
//         blockCol = ellColInd[i * cols + j];
//         for (int bi = 0; bi < ellBlockSize; ++bi)
//         {
//           rowIndex = i * ellBlockSize + bi;
//           float* rowA = rowPointers[rowIndex];
//           int srcOffset = ellBlockSize * blockCol;
//           int dstOffset = ((i * cols + j) * ellBlockSize + bi) * ellBlockSize;
//           if (blockCol != -1)
//           {
//             memcpy(&ellValue[dstOffset], &rowA[srcOffset], ellBlockSize * sizeof(float));
//           } else
//           {
//             std::fill(&ellValue[dstOffset], &ellValue[dstOffset + ellBlockSize], 0.0f);
//           }
//         }
//       }
//     }
//   };
  /*
   *
   * END LAMBDA DEFINITIONS
   *
   */

  /*
   *
   * START PROGRAM
   *
   */

  /* Get the optimal kernelSize value. It gets stored in variable ellBlockSize */
  start = INFINITY;
  ellBlockSize = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;
  divisors = nullptr; /* Initialize the pointer */
  divisorsSize = findDivisors(x, divisors);
  omp_set_num_threads(std::min(divisorsSize, omp_get_num_procs()));
  printf("divisorsSize: %d\n", divisorsSize);
  float tStart, tEnd;

  tStart = omp_get_wtime();
# pragma omp parallel shared(maxZeroBlocks, zeroCount, ellBlockSize) reduction(min:start)
  {
    int localKernelSize, localZeroBlocks, localNZeroes;
    float localStart = omp_get_wtime();
    start = localStart;
    /* Loop is reversed because we try to balance work better. NOTE: This should become
     * effective only when we are working with very big dimensions. */
#   pragma omp for schedule(guided)
    for (i = divisorsSize - 1; i >= 0; --i )
    {
      localKernelSize = divisors[i];
      localZeroBlocks = computeZeroBlocks(A, x, y, localKernelSize);
      if (localZeroBlocks > 0)
      {
        localNZeroes = localZeroBlocks * localKernelSize * localKernelSize;
          if (zeroCount < localNZeroes)
          {
#           pragma omp critical
            {
              maxZeroBlocks = localZeroBlocks;
              zeroCount = localNZeroes;
              ellBlockSize = localKernelSize;
            }
          }
      } else
      {
        continue;
      }
    }
  }
  tEnd = omp_get_wtime();
  printf("computeZeroBlocks time: %f\n", tEnd - tStart);
  omp_set_num_threads(omp_get_num_procs());

  tStart = omp_get_wtime();
  /* Now get ellCols, the actual number of columns in the BELL format */
  bSums = computeEllCols(A, x, y, ellBlockSize);
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
  getEllColInd(bSums, ellColInd, rows, cols);
  tEnd = omp_get_wtime();
  printf("getEllColInd time: %f\n", tEnd - tStart);

  tStart = omp_get_wtime();
  /* Create the ellValue array */
  ellValue = (float*) malloc((rows*cols*ellBlockSize*ellBlockSize)*sizeof(float));
  getEllValues(A, ellValue, ellColInd, rows, cols, ellBlockSize);
  tEnd = omp_get_wtime();
  printf("getEllValues time: %f\n", tEnd - tStart);
  end = omp_get_wtime();
  /*
   *
   * END PROGRAM
   *
   */
  if (PRINT_DEBUG)
  {
    std::cout << A << std::endl;
    std::cout << bSums << std::endl;
    printMat(ellColInd, rows, cols);
    printEllValue(ellValue, rows, cols, ellBlockSize);
  }
  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, ellBlockSize);
  printf("Matrix has %d zero blocks of size %d\n", maxZeroBlocks, ellBlockSize);
  printf("Total time needed for computation: %7.6f\n", end - start);

  free(divisors);
  return EXIT_SUCCESS;
}
