#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>

/* T can be int, double, float, or any numeric type */
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
/* COO matrix representation */
struct COOMatrix
{
  std::vector<int> rowidx;
  std::vector<int> colidx;
  std::vector<T> values;

  void convertMatrix(const T* X, const int m, const int n)
  {
    int i, j;
    for (i = 0; i < m; i++)
    {
      for (j = 0; j < n; j++)
      {
        T val = X[i*n + j];
        if (val != static_cast<T>(0))
        {
          rowidx.push_back(i);
          colidx.push_back(j);
          values.push_back(val);
        }
      }
    }
  }
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
/* CSR matrix representation */
struct CSRMatrix
{
  std::vector<int> rowptrs;
  std::vector<int> colidx;
  std::vector<T> values;
};


int main(int argc, char** argv)
{
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  COOMatrix<int> COO;
  int M[m * n];
  for (int i = 0; i < m*n; i++)
  {
    if (i % 2 == 0)
    {
      M[i] = 0;
    } else
    {
      M[i] = i;
    }
  }
  double start = omp_get_wtime();
  COO.convertMatrix(M, m, n);
  double end = omp_get_wtime();
  printf("Size of COO converted matrix is %ld\nTime for conversion: %lf", COO.values.size(), end - start);
  return 0;
}
