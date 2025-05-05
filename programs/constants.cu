/* Constants. ATTENTION: You MUST allocate these values with cudaMemcpyToSymbol (...) */
__constant__ int rows_d;
__constant__ int cols_d;
__constant__ int sharedMemPerBlock_d;
__constant__ int maxThreadsPerBlock_d;
__constant__ int maxThreadsPerSM_d;
__constant__ int sharedMemPerSM_d;
__constant__ int maxBlockDimSize_d;
__constant__ int totalGlobalMem_d;
__constant__ int totalConstantMem_d;
