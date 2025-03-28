#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

int bestBlockSize(torch::Tensor& A,   /* in */
                  int x,              /* in */
                  int y,              /* in */
                  int& ellBlockSize,  /* out */
                  int& ellCols,       /* out */
                  int*& ellColInd,    /* out */
                  float*& ellValue);  /* out */

void printEllValue(float* ellValue, int rows, int cols, int ellBlockSize);
int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: x, y, threshold\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int x, y, ellBlockSize, ellCols;
  float threshold, *ellValue;
  torch::Tensor A, B, bSums;
  int *ellColInd = nullptr;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  threshold = atof(argv[3]);

  A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);
  bestBlockSize(A, x, y, ellBlockSize, ellCols, ellColInd, ellValue);
  printf("BEST_KERNEL_SIZE: %d\n", ellBlockSize);
  printf("ELLCOLS: %d\n", ellCols);

  free(ellColInd);

  printf("All done. Great Success!\n");
  return EXIT_SUCCESS;
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
int bestBlockSize(torch::Tensor& A, int x, int y, int& ellBlockSize, int& ellCols, int*& ellColInd, float*& ellValue)
{
  /* Variable declarations */
  int i, j, kernelSize, zeroBlocks, maxZeroBlocks, zeroCount, nZeroes, *divisors, divisorsSize, rows, cols, size, colIdx;
  float start, end;
  torch::Tensor bSums, block;

  /* Square matrix check */
  if (x != y)
  {
    printf("Matrix must be square...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  /*
	 *
	 * START LAMBDA DEFINITIONS
	 *
	 */
  auto computeZeroBlocks = [&](int kernelSize) -> int
  {
    /* Computes the number of zero blocks of size kernelSize in matrix A */
    int nBlocksH, nBlocksW;
    torch::Tensor B, bSums;
    nBlocksH = x / kernelSize;
    nBlocksW = y / kernelSize;

    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2, 3});

    return (bSums == 0).sum().item<int>();
  };
  auto computeEllCols = [&](int kernelSize) -> torch::Tensor
  {
    /*
     * This lambda returns a tensor that contains the sum of all blocks of size kernelSize in the matrix. ATTENTION: Do
     * NOT change the return value of this lambda again... it returns a tensor just get over with it
     */
    int nBlocksH, nBlocksW;
    torch::Tensor B, bSums;
    nBlocksH = x / kernelSize;
    nBlocksW = y / kernelSize;
    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2,3});
    return bSums;
  };
  auto getEllColInd = [&](torch::Tensor bSums, int* ellColInd) -> void
  {
  /*
   * Compute the size of ellColInd array, i.e. the array that stoellBlockSize the index of the column position of all non-zero
   * elements in A. If a row has < ellCols non-zero elements, the remaining elements will be set to -1
   */
    int i, j, idx, rowSize;
    float val;

    for (i = 0; i < rows; i++)
    {
      idx = 0;
      rowSize = 0;
      int *row = nullptr;
      for (j = 0; j < cols; j++)
      {
        val = bSums[i][j].item<float>();
        if (val != 0)
        {
          rowSize += 1;
          row = (int*) realloc(row, rowSize * sizeof(int));
          row[idx] = j;
          idx++;
        }
      }
      for (j = 0; j < cols; j++)
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
  };
  auto getEllValues = [&](float *ellValue, int *ellColInd)
  {
    /* This lambda gives the blocked ellpack values array */
    int nBlocksW = A.size(1) / ellBlockSize;
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        int blockCol = ellColInd[i * cols + j];
        for (int bi = 0; bi < ellBlockSize; bi++)
        {
          for (int bj = 0; bj < ellBlockSize; bj++)
          {
            int dstIndex = ((i * cols + j) * ellBlockSize + bi) * ellBlockSize + bj;
            if (blockCol != -1)
            {
              int rowIndex = i * ellBlockSize + bi;
              int colIndex = blockCol * ellBlockSize + bj;
              ellValue[dstIndex] = A[rowIndex][colIndex].item<float>();
            } else
            {
              ellValue[dstIndex] = 0.0f;
            }
          }
        }
      }
    }
  };
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
  start = omp_get_wtime();
  ellBlockSize = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;
  divisors = nullptr; /* Initialize the pointer */
  divisorsSize = findDivisors(x, divisors);

  for (i = 0; i < divisorsSize; i++)
  {
    kernelSize = divisors[i];
    zeroBlocks = computeZeroBlocks(kernelSize);
    if (zeroBlocks == 0)
    {
      break;
    }
    nZeroes = zeroBlocks * kernelSize * kernelSize;
    if (zeroCount <= nZeroes)
    {
      maxZeroBlocks = zeroBlocks;
      zeroCount = nZeroes;
      ellBlockSize = kernelSize;
    }
  }

  /* Now get ellCols, the actual number of columns in the BELL format */
  bSums = computeEllCols(ellBlockSize);
  ellCols = (bSums != 0).sum(1).max().item<int>();

  /* Allocate memory for ellColInd array */
  cols = ellCols;
  rows = x / ellBlockSize;
  size = rows*cols;
  ellColInd = (int*) malloc(size*sizeof(int));

  /* Create the ellColInd array */
  getEllColInd(bSums, ellColInd);


  // printTensor(bSums, bSums.size(0), bSums.size(1));
  printMat(ellColInd, rows, cols);
  std::cout << A << std::endl;
  ellValue = (float*) malloc((rows*cols*ellBlockSize*ellBlockSize)*sizeof(float));
  getEllValues(ellValue, ellColInd);

  end = omp_get_wtime();
  /*
   *
   * END PROGRAM
   *
   */

  printEllValue(ellValue, rows, cols, ellBlockSize);

  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, ellBlockSize);
  printf("Matrix has %d zero blocks of size %d\n", maxZeroBlocks, ellBlockSize);
  printf("Total time needed for computation: %f\n", end - start);

  free(divisors);
  // free(ellColInd);
  free(ellValue);
  return EXIT_SUCCESS;
}


