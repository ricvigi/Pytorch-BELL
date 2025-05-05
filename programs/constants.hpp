#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

__constant__ extern int rows_d;
__constant__ extern int cols_d;
__constant__ extern int sharedMemPerBlock_d;
__constant__ extern int maxThreadsPerBlock_d;
__constant__ extern int maxThreadsPerSM_d;
__constant__ extern int sharedMemPerSM_d;
__constant__ extern int maxBlockDimSize_d;
__constant__ extern int totalGlobalMem_d;
__constant__ extern int totalConstantMem_d;

#endif
