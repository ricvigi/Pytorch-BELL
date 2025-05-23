/*
 * ---------------------------------------------------------------------------------------
 * File        : cudaFindBellParams.cu
 * License     : MIT License (see LICENSE file)
 *
 * License Summary : THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 *                   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *                   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *                   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *                   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *                   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *                   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Description:
 * ----------------------------------------------------------------------------------------
 */
#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include "myHeaders.hpp"
#include "cudaHeaders.hpp"

int PRINT_DEBUG = 0;


int main (int argc, char** argv)
{
  if (argc < 5)
  {
    printf("Usage: x, y, threshold, print debug\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }
  int rows, cols;
  float threshold;
  torch::Tensor A;

  /* rows and cols are placed in constant memory in the device */
  rows = atoi(argv[1]);
  cols = atoi(argv[2]);
  threshold = atof(argv[3]);
  PRINT_DEBUG = atoi(argv[4]);


  A = torch::randn({rows, cols});
  A.masked_fill_(A < threshold, 0);


  /* ATTENTION: do NOT remove this function call */
  cGetDeviceProp();

  launch_cGetBellParams(A, rows, cols, 2);
  return EXIT_SUCCESS;
}
