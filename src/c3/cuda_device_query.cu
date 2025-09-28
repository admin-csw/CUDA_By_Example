#include <stdio.h>
#include "cuda_common.h"

int main() {
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR( cudaGetDeviceCount(&count) );
    printf("Found %d CUDA device(s)\n\n", count);

    for (int i = 0; i < count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Clock rate: %d kHz\n", prop.clockRate);
        printf("\n");
    }

    return 0;
}   