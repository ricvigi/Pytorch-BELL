/*
 * ---------------------------------------------------------------------------------------
 * File        : functions.cu
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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>             // printf
#include <iostream>
#include <torch/torch.h>
#include <omp.h>
#include "cudaHeaders.hpp"
#include "myHeaders.hpp"

__global__
void cGetBellParams(torch::Tensor& A, int x, int y, int& ellBlockSize, int& ellCols, int*& ellColInd, float*& ellValue)
{
  /* device variables */
  float *A_d, *ellValue_d;
  int *ellBlockSize_d, *ellCols_d, *ellColInd_d;

  int A_size = A.numel();

  /* Get a pointer to the tensor */
  float *A_h = A.contiguous().data_ptr<float>();

  /* Device memory allocation */
  CHECK_CUDA( cudaMalloc((void**) &A_d, A_size * sizeof(float)) )
  CHECK_CUDA( cudaMemcpy(A_d, A_h, A_size * sizeof(float), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMalloc((void**) &ellBlockSize_d, sizeof(int), cudaMemcpyHostToDevice) )
}

/**
 * @brief Function that prints desired device properties
 *
 * @return void
 */
__global__
void cGetDeviceProp()
{
  cudaDeviceProp prop;
	CHECK_CUDA_CALL( cudaGetDevice(0) );
	CHECK_CUDA_CALL( cudaGetDeviceProperties(&prop, 0) );
	printf("Max shared memory: %lu\n", prop.sharedMemPerBlock);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "true" : "false");
	printf("32 bit registers per SM: %d\n", prop.regsPerMultiprocessor);
	printf("Shared memory per SM: %lu\n", prop.sharedMemPerMultiprocessor);
}


__device__
int cIterativeComputeZeroBlocks(float* A, int rows, int cols, int kernelSize)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;

	/* id of a thread inside a block */
	int blockId = threadIdx.x;
}

