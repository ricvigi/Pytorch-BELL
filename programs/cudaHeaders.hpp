/*
 * ---------------------------------------------------------------------------------------
 * File        : cudaHeaders.hpp
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
 * Description: This is a header file that contains declarations and some utility functions
 * ----------------------------------------------------------------------------------------
 */


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
        exit(EXIT_FAILURE);                                                    \
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
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}
#endif



__global__
void cGetBellParams(float* A_d,             /* in */
                    int x,                  /* in */
                    int y,                  /* in */
                    int A_size,             /* in */
                    int* ellBlockSize_d,    /* out */
                    int* ellCols_d,         /* out */
                    int* ellColInd_d,       /* out */
                    float* ellValue_d       /* out */);
__device__
int cIterativeComputeZeroBlocks(float* A,       /* in */
                                int rows,       /* in */
                                int cols,       /* in */
                                int kernelSize  /* in */);

void cGetDeviceProp();


#endif
