#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include "my_headers.hpp"

const int EXIT_UNSUPPORTED = 2;


