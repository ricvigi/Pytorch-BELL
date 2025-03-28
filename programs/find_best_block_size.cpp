#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

int bestBlockSize(torch::Tensor& A, /* in */
                  int x,            /* in */
                  int y,            /* in */
                  int& res,         /* out */
                  int& ellCols);    /* out */

void printEllValue(float* ellValue, int rows, int cols, int res);
int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: x, y, threshold\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int x, y, res, ellCols;
  float threshold;
  torch::Tensor A, B, bSums;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  threshold = atof(argv[3]);

  A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);
  bestBlockSize(A, x, y, res, ellCols);
  printf("BEST_KERNEL_SIZE: %d\n", res);
  printf("ELLCOLS: %d\n", ellCols);


  printf("All done. Great Success!\n");
  return EXIT_SUCCESS;
}

/**
 * @brief Find the block size that filters out the most zeroes
 *
 * @param &A Reference to sparse tensor that we want to convert to BELL
 * @param x Size of the first dimension of A
 * @param y Size of the second dimension of A
 * @param &res Reference to the variable that will store the best block size for tensor A
 * @param &ellCols Reference to the variable that will store the number of columns in BELL
 *
 * @return Best block size on success, EXIT_FAILURE on error.
 */
int bestBlockSize(torch::Tensor& A, int x, int y, int& res, int& ellCols)
{
  /* Variable declarations */
  int i, j, kernelSize, nBlocksH, nBlocksW, zeroBlocks, maxZeroBlocks, zeroCount, nZeroes, *divisors, divisorsSize, *ellColInd, rows, cols, size, colIdx;
  float start, end, *ellValue;
  torch::Tensor B, bSums, block;

  /* Square matrix check */
  if (x != y)
  {
    printf("We are only accepting square matrices for now...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  /* Lambdas */
  auto computeZeroBlocks = [&](int kernelSize) -> int
  {
    /* Computes the number of zero blocks of size kernelSize in matrix A */
    int nBlocksH = x / kernelSize;
    int nBlocksW = y / kernelSize;

    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2, 3});

    return (bSums == 0).sum().item<int>();
  };
  auto computeEllCols = [&](int kernelSize) -> int
  {
    /* This lambda returns a tensor that contains the sum of all blocks of size kernelSize */
    nBlocksH = x / kernelSize;
    nBlocksW = y / kernelSize;
    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2,3});
    // printTensor(bSums, nBlocksH, nBlocksW);
    return (bSums != 0).sum(1).max().item<int>();
  };
  auto getEllColInd = [&]() -> void
  {
    int i, j, idx, rowSize;
    float val;

    for (i = 0; i < bSums.size(0); i++)
    {
      idx = 0;
      rowSize = 0;
      int *row = nullptr; /* This need to be nullptr because we use realloc in the loop */
      for (j = 0; j < bSums.size(1); j++)
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
  auto fillEllValues = [&]()
  {
    int nBlocksW = A.size(1) / res;

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        int blockCol = ellColInd[i * cols + j];
        for (int bi = 0; bi < res; bi++)
        {
          for (int bj = 0; bj < res; bj++)
          {
            int dstIndex = ((i * cols + j) * res + bi) * res + bj;
            if (blockCol != -1)
            {
              int rowIndex = i * res + bi;
              int colIndex = blockCol * res + bj;
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

  /* Get the optimal kernelSize value */
  start = omp_get_wtime();
  res = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;
  divisors = (int*) malloc(sizeof(int)); /* Initialize the pointer */
  divisorsSize = findDivisors(x, &divisors);

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
      res = kernelSize;
    }
  }

  /* Now get ellCols, the actual number of columns in the BELL format */
  /* TODO: Enclose this operation in a function */
  ellCols = computeEllCols(res);
  cols = ellCols;
  rows = bSums.size(0);
  size = rows*cols;
  ellColInd = (int*) malloc(size*sizeof(int));

  getEllColInd();
  // printTensor(bSums, bSums.size(0), bSums.size(1));
  // printMat(ellColInd, rows, cols);
  // std::cout << A << std::endl;
  ellValue = (float*) malloc((rows*cols*res*res)*sizeof(float));
  fillEllValues();

  end = omp_get_wtime();


  // printEllValue(ellValue, rows, cols, res);

  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, res);
  printf("Matrix has %d zero blocks of size %d\n", maxZeroBlocks, res);
  printf("Total time needed for computation: %f\n", end - start);

  free(divisors);
  free(ellColInd);
  free(ellValue);
  return EXIT_SUCCESS;
}


