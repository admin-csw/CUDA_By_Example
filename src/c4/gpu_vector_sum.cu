#include <stdio.h>
#include "cuda_common.h"

#define N 10000

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));    

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // display the first 10 results
    printf("First 100 results:\n");
    for (int i = 0; i < 100; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Verify a few more results to ensure correctness
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            break;
        }
    }
    printf("\nVector addition %s!\n", success ? "successful" : "failed");

    // free the memory allocated on the GPU
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c)); 

    return 0;
}