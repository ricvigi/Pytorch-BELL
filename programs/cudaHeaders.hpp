#ifndef CUDA_HEADERS
#define CUDA_HEADERS
extern int PRINT_DEBUG;

__global__
void getBellParams(torch::Tensor& A,         /* in */
                   int x,                    /* in */
                   int y,                    /* in */
                   int& ellBlockSize,        /* out */
                   int& ellCols,             /* out */
                   int*& ellColInd,          /* out */
                   float*& ellValue          /* out */);


#endif
