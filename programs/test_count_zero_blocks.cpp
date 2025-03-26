#include <torch/torch.h>
#include <omp.h>
#include <iostream>


int main(int argc, char** argv)
{
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  float threshold = atof(argv[3]);
  torch::Tensor A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);


  printf("All done\n");
  return EXIT_SUCCESS;
}
