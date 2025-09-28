// CUDA helper macros and common includes
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// HANDLE_ERROR macro - detects errors, prints them, and exits with EXIT_FAILURE
#define HANDLE_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Get GPU properties
void print_gpu_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        printf("  Shared memory per block: %.0f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max grid size: %d x %d x %d\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("\n");
    }
}

// Timer utility
struct Timer {
    cudaEvent_t start, stop;
    
    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void start_timer() {
        cudaEventRecord(start);
    }
    
    float stop_timer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};

#endif // CUDA_COMMON_H