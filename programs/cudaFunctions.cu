/*
 * ---------------------------------------------------------------------------------------
 * File        : cudaFunctions.cu
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
#include <cstdio>
#include <iostream>
#include <torch/torch.h>
#include <omp.h>
#include "cudaHeaders.hpp"
#include "myHeaders.hpp"

// /* Constants. ATTENTION: You MUST allocate these values with cudaMemcpyToSymbol (...) */
__constant__ int rows_d;
__constant__ int cols_d;
__constant__ int sharedMemPerBlock_d;
__constant__ int maxThreadsPerBlock_d;
__constant__ int maxThreadsPerSM_d;
__constant__ int sharedMemPerSM_d;
__constant__ int maxBlockDimSize_d;
__constant__ int totalGlobalMem_d;
__constant__ int totalConstantMem_d;


/**
 * @brief Prepares the environment and lanches the kernel cGetBellParams
 *
 * @param &A Reference to the tensor to be transformed
 * @param rows First dimension of A
 * @param cols Second dimension of A
 *
 * @return an integer representing success or a failure
 */
int launch_cGetBellParams (torch::Tensor& A, int rows, int cols)
{
  float *A_d, *A_h;
  int A_size;

  if (rows != cols)
  {
    printf("Matrix must be square\n");
    fflush(stdout);
    return EXIT_FAILURE;
  } else if (isPrime(rows))
  {
    printf("Matrix dimensions can't be prime\n");
    fflush(stdout);
    return EXIT_FAILURE;
  }

  /* ATTENTION: Change these values */
  dim3 dimBlock(1);
  dim3 dimGrid(1);

  A_size = rows * cols;
  int* divisors = nullptr;
  int divisorsSize = findDivisors(rows, divisors);

  A_h = A.contiguous().data_ptr<float>();
  CHECK_CUDA( cudaMalloc((void**) &A_d, A_size * sizeof(float)) );
  CHECK_CUDA( cudaMemcpy(A_d, A_h, A_size * sizeof(float), cudaMemcpyHostToDevice) );

  return EXIT_SUCCESS;
}

/**
 * @brief Extract the parameters needed by cuSPARSE API to construct a BLOCKED-ELL object
 *
 * @param *A_d Pointer to the desired tensor in GPU memory.
 * @param *ellBlockSize_d Pointer to integer that will stored the best block size.
 * @param *ellCols_d Pointer to integer representing the maximum number of (ellBlockSize_d) columns in ellValue_d array.
 * @param *ellColInd_d Pointer to array of integers containg indices of the blocked-ell array.
 * @param *ellValue_d Pointer to values array after reordering.
 *
 * @return void
 */
__global__
void cGetBellParams (float *A_d, int *ellBlockSize_d, int *ellCols_d, int *ellColInd_d, float *ellValue_d)
{
  /* TODO: Complete this... */
}

/**
 * @brief Function that prints desired device properties
 *
 * @return void
 */
void cGetDeviceProp ()
{
  cudaDeviceProp prop;
  int deviceID = 0;
  CHECK_CUDA( cudaGetDevice(&deviceID) );
  CHECK_CUDA( cudaGetDeviceProperties(&prop, deviceID) );
  printf("Device name: %s\n", prop.name);
  printf("Max shared memory: %lu\n", prop.sharedMemPerBlock);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "true" : "false");
  printf("32 bit registers per SM: %d\n", prop.regsPerMultiprocessor);
  printf("Shared memory per SM: %lu\n", prop.sharedMemPerMultiprocessor);
  printf("MultiGPU board: %s\n", prop.isMultiGpuBoard ? "true" : "false");
  printf("Max size of each grid dimension: %d\n", *prop.maxGridSize);
  printf("Max size of each dimension of a block: %d\n", *prop.maxThreadsDim);
  printf("Total global memory: %ld\n", (long) prop.totalGlobalMem);
  printf("Total constant memory: %ld\n", (long) prop.totalConstMem);
  printf("Unified addressing: %s\n", prop.unifiedAddressing ? "true" : "false");

  /* Allocate device properties in constant memory so that you apply the same routines to different GPUs */
  // CHECK_CUDA( cudaMemcpyToSymbol(sharedMemPerBlock_d, &prop.sharedMemPerBlock, sizeof(int)) )
  // CHECK_CUDA( cudaMemcpyToSymbol(maxThreadsPerBlock_d, &prop.maxThreadsPerBlock, sizeof(int)) )
  // CHECK_CUDA( cudaMemcpyToSymbol(maxThreadsPerSM_d, &prop.maxThreadsPerMultiProcessor, sizeof(int)) )
  // CHECK_CUDA( cudaMemcpyToSymbol(sharedMemPerSM_d, &prop.sharedMemPerMultiprocessor, sizeof(int)) )
  // CHECK_CUDA( cudaMemcpyToSymbol(maxBlockDimSize_d, &prop.maxThreadsDim, sizeof(int)) )
  // CHECK_CUDA( cudaMemcpyToSymbol(totalGlobalMem_d, (long*) prop.totalGlobalMem, sizeof(long)) ) /* ATTENTION: This call introduces an error */
  // CHECK_CUDA( cudaMemcpyToSymbol(totalConstantMem_d, (long*) prop.totalConstMem, sizeof(long)) ) /* ATTENTION: This call introduces an error */
}

__device__
int cIterativeComputeZeroBlocks (float* A_d, int kernelSize)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;

  /* id of a thread inside a block */
  int blockId = threadIdx.x;

  extern __shared__ int sMem[];
  return 0;
}

