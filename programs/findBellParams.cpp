#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "myHeaders.hpp"

int PRINT_DEBUG = 0;

int main(int argc, char** argv)
{

  if (argc < 5)
  {
    printf("Usage: x, y, threshold, print debug\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int x, y, ellBlockSize, ellCols, err;
  float threshold;
  torch::Tensor A, B, bSums;

  int *ellColInd = nullptr;
  float *ellValue = nullptr;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  threshold = atof(argv[3]);
  PRINT_DEBUG = atoi(argv[4]);

  torch::manual_seed(42);

  A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);
  err = getBellParams(A, x, y, ellBlockSize, ellCols, ellColInd, ellValue);
  if (err != 0)
  {
    printf("Error code %d, exiting!\n", err);
    fflush(stdout);
    return err;
  }
  printf("BEST_KERNEL_SIZE: %d\n", ellBlockSize);
  printf("ELLCOLS: %d\n", ellCols);

  free(ellColInd);
  free(ellValue);
  printf("All done. Great Success!\n");
  return EXIT_SUCCESS;
}



