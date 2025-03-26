#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include "myHeaders.hpp"

int main(int argc, char** argv)
{
  if (argc < 4)
  {
    printf("Usage: x, y, threshold, kernelSize\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int x, y, kernelSize, nBlocksH, nBlocksW, zeroBlocks;
  float threshold, start, end;
  torch::Tensor A, B, bSums;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  threshold = atof(argv[3]);
  kernelSize = atoi(argv[4]);

  if (x != y)
  {
    printf("We are only accepting square matrices for now...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  } else if (((x % kernelSize) != 0) || (y % kernelSize) != 0)
  {
    printf("Matrix size %d must be evenly divisible by kernelSize %d\n", x, kernelSize);
    fflush(stdout);
    return EXIT_FAILURE;
  }


  A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);

  start = omp_get_wtime();
  nBlocksH = x / kernelSize;
  nBlocksW = y / kernelSize;

  B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
  printf("Matrix shape is %ld %ld %ld %ld\n", B.size(0), B.size(1), B.size(2), B.size(3));
  B = B.permute({0, 2, 1, 3});
  bSums = B.sum({2, 3});
  zeroBlocks = (bSums == 0).sum().item<int>();

  end = omp_get_wtime();
  printf("A contains %d zero blocks of size %d\n", zeroBlocks, kernelSize);
  printf("Total time needed for computation: %f\n", end - start);


  printf("All done. Great Success!\n");
  return EXIT_SUCCESS;
}
