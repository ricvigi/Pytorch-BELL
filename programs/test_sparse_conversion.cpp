#include <torch/torch.h>
#include <omp.h>
#include <iostream>

void print_tensor(torch::Tensor *A, int x, int y);

int main(int argc, char** argv)
{
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  float threshold = atof(argv[3]);
  torch::Tensor A = torch::randn({x, y});
  A.masked_fill_(A < threshold, 0);
  // print_tensor(&A, x, y);

  float start = omp_get_wtime();
  torch::Tensor A_csr = A.to_sparse_csr();
  float end = omp_get_wtime();
  // std::cout << "CSR Tensor: " << A_csr << std::endl;
  // std::cout << "Crow indices: " << A_csr.crow_indices() << std::endl;
  // std::cout << "Col indices: " << A_csr.col_indices() << std::endl;
  // std::cout << "Values: " << A_csr.values() << std::endl;
  std::cout << "Elapsed time: " << end - start << std::endl;

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
