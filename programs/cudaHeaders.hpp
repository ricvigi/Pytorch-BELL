#ifndef CUDA_HEADERS
#define CUDA_HEADERS
extern int PRINT_DEBUG;

#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#endif



__global__
void cGetBellParams(torch::Tensor& A,        /* in */
                   int x,                    /* in */
                   int y,                    /* in */
                   int& ellBlockSize,        /* out */
                   int& ellCols,             /* out */
                   int*& ellColInd,          /* out */
                   float*& ellValue          /* out */);
__device__
int cIterativeComputeZeroBlocks(float* A,       /* in */
                                int rows,       /* in */
                                int cols,       /* in */
                                int kernelSize  /* in */);


#endif
