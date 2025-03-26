#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Usage: x, y, threshold\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int x, y, kernelSize, nBlocksH, nBlocksW, zeroBlocks, maxZeroBlocks, res, zeroCount, temp;
  float threshold, start, end;
  torch::Tensor A, B, bSums;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  threshold = atof(argv[3]);
  kernelSize = 2;
  if (x != y)
  {
    printf("We are only accepting square matrices for now...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);
  // This matrix is used to test the ability of the algorithm to prefer larger kernel sizes
  // A = torch::tensor({
  //       {0, 0, 0, 0, 1, 1, 1, 1},
  //       {0, 0, 0, 0, 1, 1, 1, 1},
  //       {0, 0, 0, 0, 1, 1, 1, 1},
  //       {0, 0, 0, 0, 1, 1, 1, 1},
  //       {1, 1, 1, 1, 0, 0, 0, 0},
  //       {1, 1, 1, 1, 0, 0, 0, 0},
  //       {1, 1, 1, 1, 0, 0, 0, 0},
  //       {1, 1, 1, 1, 0, 0, 0, 0}
  //   });
  res = 0;
  zeroCount = 0;
  maxZeroBlocks = 0;

  start = omp_get_wtime();
  do
  {
    printf("Kernel size: %d\n", kernelSize);
    nBlocksH = x / kernelSize;
    nBlocksW = y / kernelSize;
    B = A.view({nBlocksH, kernelSize, nBlocksW, kernelSize});
    B = B.permute({0, 2, 1, 3});
    bSums = B.sum({2, 3});
    zeroBlocks = (bSums == 0).sum().item<int>();
    temp = zeroBlocks * kernelSize * kernelSize; /* ATTENTION: To correctly compute the number of zeroes you are filtering out, you have to multiply the number of zero blocks with the number of elements in the kernel! This is because bigger kernels will tend to have less zero blocks than smaller ones, but might actually filter out more zeroes! */
    if (zeroCount <= temp) /* <= condition should ensure that we always select a bigger kernel size if two sizes yield the same results. ATTENTION: check if this is actually true */
    {
      maxZeroBlocks = zeroBlocks;
      zeroCount = temp;
      res = kernelSize;
    }
    printf("We can filter out %d zeroes with a kernel of size %d\n", temp, kernelSize);
    printf("Matrix has %d zero blocks of size %d\n", zeroBlocks, kernelSize);

    kernelSize *= 2;
  } while (kernelSize <= floor(x / 2));

  end = omp_get_wtime();


  printf("We can filter out %d zeroes with a kernel of size %d\n", zeroCount, res);
  printf("Matrix has %d zero blocks of size %d\n", maxZeroBlocks, res);
  printf("Total time needed for computation: %f\n", end - start);


  printf("All done. Great Success!\n");
  return EXIT_SUCCESS;
}
