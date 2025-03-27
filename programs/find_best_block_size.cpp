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
  int i, kernelSize, nBlocksH, nBlocksW, zeroBlocks, maxZeroBlocks, zeroCount, nZeroes, *divisors, divisorsSize;
  float start, end;
  torch::Tensor B, bSums;
  if (x != y)
  {
    printf("We are only accepting square matrices for now...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  res = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;
  divisors = (int*) malloc(sizeof(int)); /* Initialize the pointer */
  divisorsSize = findDivisors(x, &divisors);

  // printTensor(A, x, y);

  /* Lambdas */
  /* Computes the number of zero blocks of size kernelSize in matrix A */
  auto computeZeroBlocks = [&](int kernelSize) -> int
  {
    // printf("Kernel size: %d\n", kernelSize);

    int nBlocksH = x / kernelSize;
    int nBlocksW = y / kernelSize;

    torch::Tensor B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    torch::Tensor bSums = B.sum({2, 3});

    return (bSums == 0).sum().item<int>();
  };

  auto computeEllCols = [&](int kernelSize) -> torch::Tensor /* This lambda returns a tensor because we need it in another lambda */
  {
    nBlocksH = x / kernelSize;
    nBlocksW = y / kernelSize;
    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    // printTensor(bSums, nBlocksH, nBlocksW);
    return B.sum({2, 3});
  };
  auto getEllColInd = [&](torch::Tensor M, int x, int ellCols, int kernelSize) -> int*
  {/* TODO: finish this... */
    int *ellColInt, rows, cols, size;
    rows = ellCols / kernelSize;
    cols = x / kernelSize;
    size = rows*cols;
    ellColInt = (int*) malloc(size * sizeof(int));


  }

  /* Get the optimal kernelSize value */
  start = omp_get_wtime();
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
    // printf("We can filter out %d zeroes with a kernel of size %d\n", nZeroes, kernelSize);
    // printf("Matrix has %d zero blocks of size %d\n", zeroBlocks, kernelSize);

  }

  /* Now get ellCols, the actual number of columns in the BELL format */
  /* TODO: Enclose this operation in a function */
  bSums = computeEllCols(res);
  ellCols = (bSums != 0).sum(1).max().item<int>();

  end = omp_get_wtime();

  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, res);
  printf("Matrix has %d zero blocks of size %d\n", maxZeroBlocks, res);
  printf("Total time needed for computation: %f\n", end - start);

  free(divisors);
  return EXIT_SUCCESS;
}
