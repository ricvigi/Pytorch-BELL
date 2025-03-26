#include <torch/torch.h>
#include <omp.h>
#include <iostream>

void print_tensor(torch::Tensor *A, int x, int y);

int main(int argc, char** argv)
{
  if (argc < 4)
  {
    printf("Usage: x, y, threshold, kernel_size\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  float threshold = atof(argv[3]);
  int kernel_size = atoi(argv[4]);
  if (x != y)
  {
    printf("We are only accepting square matrices for now...\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  if ((kernel_size % x) != 0)
  {
    printf("Matrix size %d must be evenly divisible by kernel_size %d\n", x, kernel_size);
    fflush(stdout);
    return EXIT_FAILURE;
  }
  torch::Tensor A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);

  n_blocks_x = A.size(0) / kernel_size;


  printf("All Good\n");
  return EXIT_SUCCESS;
}



void print_tensor(torch::Tensor *A, int x, int y)
{
  for (int i = 0; i < x; i++)
  {
    for (int j = 0; j < y; j++)
    {
      printf("%f ", (*A).index({i, j}).item<float>());
    }
    printf("\n");
  }
}
