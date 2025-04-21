/*
 * ---------------------------------------------------------------------------------------
 * File        : functions.cpp
 * License     : MIT License (see LICENSE file)
 *
 * License Summary : THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 *                   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *                   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *                   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *                   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *                   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *                   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Description: This file contains functions whose purpose is to extract the necessary
 *              parameters that can be passed to the cuSPARSE API in order to create a
 *              BLOCKED-ELL sparse matrix object.
 * ----------------------------------------------------------------------------------------
 */

#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

/**
 * @brief Computes the number of zero blocks of size kernelSize in tensor A
 *
 * @param &A Reference to the tensor
 * @param rows Size of the first dimension of A
 * @param cols Size of the second dimension of A
 * @param kernelSize Size of the blocks (square)
 *
 * @return int, the number of zero blocks of size kernelSize x kernelSize in tensor A
 */
int computeZeroBlocks(torch::Tensor &A, int rows, int cols, int kernelSize)
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
int iterativeComputeZeroBlocks(torch::Tensor &A, int rows, int cols, int kernelSize)
{
  int count = 0;
  int nBlocksH = rows / kernelSize;
  int nBlocksW = cols / kernelSize;
  std::vector<float*> rowPointers (rows);
# pragma omp parallel
  {
#   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<float>();
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
            if (rowPointers[row][col] != 0.0f)
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
torch::Tensor computeEllCols(torch::Tensor& A, int rows, int cols, int kernelSize)
{
  int nBlocksH, nBlocksW;
  torch::Tensor B, bSums;
  nBlocksH = rows / kernelSize;
  nBlocksW = cols / kernelSize;
  B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
  B = B.permute({0, 2, 1, 3});
  bSums = B.sum({2,3});
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
torch::Tensor iterativeComputeEllCols(torch::Tensor& A, int rows, int cols, int kernelSize)
{
  int nBlocksH = rows / kernelSize;
  int nBlocksW = cols / kernelSize;
  torch::Tensor bSums;
  float* tBSums;
  tBSums = (float*) malloc(rows*cols*sizeof(float));
  std::vector<float*> rowPointers (rows);
  auto del = [](void* ptr) { free(ptr); };

# pragma omp parallel
  {
#   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<float>();
    }
#   pragma omp for collapse(2)
    for (int i = 0; i < nBlocksH; ++i)
    {
      for (int j = 0; j < nBlocksW; ++j)
      {
        float sum = 0.0f;
        for (int ii = 0; ii < kernelSize; ++ii)
        {
          for (int jj = 0; jj < kernelSize; ++jj)
          {
            int row = i * kernelSize + ii;
            int col = j * kernelSize + jj;
            float val = rowPointers[row][col];
            sum += val;
          }
        }
        tBSums[i * kernelSize + j] = sum;
      }
    }
  }
  // bSums = torch::from_blob(tBSums, {rows, cols}, torch::kFloat32).clone();
  // free(tBSums);
  bSums = torch::from_blob(tBSums, {rows, cols}, del, torch::TensorOptions().dtype(torch::kFloat32)
);
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
void getEllColInd(torch::Tensor &bSums, int *ellColInd, int rows, int cols)
{
  int idx, rowSize;
  float val;

  std::vector<float*> rowPointers(rows);
# pragma omp parallel shared(ellColInd) private(idx, rowSize, val)
  {
#   pragma omp for
    for (int i = 0; i < rows; ++i)
    {
      rowPointers[i] = bSums[i].contiguous().data_ptr<float>();
    }

#   pragma omp for
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
void getEllValues(torch::Tensor& A, float *ellValue, int *ellColInd, int rows, int cols, int ellBlockSize)
{
  int blockCol, dstIndex, rowIndex, colIndex;
  std::vector<float*> rowPointers(rows * ellBlockSize);

# pragma omp parallel shared(rowPointers, A, ellColInd, ellValue) private(blockCol, dstIndex, rowIndex, colIndex)
  {
#   pragma omp for
    for (int i = 0; i < rows * ellBlockSize; ++i)
    {
      rowPointers[i] = A[i].contiguous().data_ptr<float>();
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
  int i, j, kernelSize, zeroBlocks, maxZeroBlocks, zeroCount, nZeroes, *divisors, divisorsSize, rows, cols, size, colIdx, nThreads;
  float start, end, tStart, tEnd;
  torch::Tensor bSums;

  /*
   * ATTENTION: This instruction allows nested parallelism. Right now it improves performance, but i'm
   * not sure why :/...
   * If you remove this, or set it to 0, all calls to functions that contain parallel regions, if called
   * inside a parallel region, will be executed sequentially
   */
  omp_set_nested(1);

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
   * START PROGRAM
   *
   */

  /* The optimal block size shape will be stored in ellBlockSize. */
  ellCols = 0;
  ellBlockSize = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;
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
    int localKernelSize, localZeroBlocks, localNZeroes, localBestZeroes, localBestKernel, id;
    id = omp_get_thread_num();
    localBestZeroes = 0;

    /* NOTE: Should we use a guided schedule for this loop parallelism? */
#   pragma omp for
    for (i = 0; i < divisorsSize; ++i)
    {
      localKernelSize = divisors[i];
      localZeroBlocks = iterativeComputeZeroBlocks(A, x, y, localKernelSize);
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
  bSums = computeEllCols(A, x, y, ellBlockSize);
  // bSums = iterativeComputeEllCols(A, x, y, ellBlockSize);
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
  // getEllValues(A, ellValue, ellColInd);
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
  printf("Matrix has %d zero blocks of size %d\n", zeroCount / (ellBlockSize*ellBlockSize), ellBlockSize);
  printf("Total time needed for computation: %7.6f\n", end - start);

  free(divisors);
  return EXIT_SUCCESS;
}
