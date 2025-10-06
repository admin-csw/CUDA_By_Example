#include "cuda.h"
#include "cuda_common.h"
#include <stdio.h>

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/period) + 1.0f) * 
                                            (sinf(y*2.0f*PI/period) + 1.0f) / 4.0f;

    __syncthreads();
    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

void save_ppm(unsigned char *bitmap, int width, int height, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        // Convert RGBA to RGB for PPM (use green channel for pattern)
        unsigned char r = 0;
        unsigned char g = bitmap[i * 4 + 1];  // Green channel has the pattern
        unsigned char b = 0;
        fwrite(&r, 1, 1, file);
        fwrite(&g, 1, 1, file);
        fwrite(&b, 1, 1, file);
    }
    
    fclose(file);
    printf("Shared memory pattern saved to %s\n", filename);
}

int main() {
    unsigned char *bitmap;
    unsigned char *dev_bitmap;
    int image_size = DIM * DIM * 4; // RGBA
    
    // Allocate memory
    bitmap = (unsigned char*)malloc(image_size);
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, image_size));
    
    printf("Generating GPU Shared Memory Pattern (%dx%d)...\n", DIM, DIM);

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    kernel<<<grids, threads>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap, dev_bitmap, image_size, cudaMemcpyDeviceToHost));
    
    save_ppm(bitmap, DIM, DIM, "shared_memory_pattern.ppm");
    
    printf("Shared memory demonstration complete!\n");
    printf("Pattern shows sine wave interference using shared memory for cross-thread communication.\n");
   
    // Cleanup
    free(bitmap);
    HANDLE_ERROR(cudaFree(dev_bitmap));
    
    return 0;
}