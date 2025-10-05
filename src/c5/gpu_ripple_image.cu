#include "cuda_common.h"
#include <stdio.h>

#define DIM 1024

__global__ void kernel(unsigned char *ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * gridDim.x * blockDim.x;

    // now calculate the value at that position
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                       cos(d / 10.0f - ticks / 7.0f) /
                                       (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

void save_ppm(unsigned char *bitmap, int width, int height, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return;
    }
    
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        // Convert RGBA to RGB for PPM
        fwrite(&bitmap[i * 4], 3, 1, file);
    }
    
    fclose(file);
    printf("Ripple effect saved to %s\n", filename);
}

int main(void) {
    unsigned char *dev_bitmap;
    unsigned char *bitmap;
    int image_size = DIM * DIM * 4; // RGBA
    
    // Allocate memory
    bitmap = (unsigned char*)malloc(image_size);
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, image_size));
    
    printf("Generating GPU Ripple Effect (%dx%d)...\n", DIM, DIM);
    
    // Generate ripple effect at different time steps
    for (int ticks = 0; ticks < 100; ticks += 20) {
        dim3 blocks(DIM / 16, DIM / 16);
        dim3 threads(16, 16);
        
        kernel<<<blocks, threads>>>(dev_bitmap, ticks);
        HANDLE_ERROR(cudaMemcpy(bitmap, dev_bitmap, image_size, cudaMemcpyDeviceToHost));
        
        char filename[50];
        sprintf(filename, "ripple_frame_%02d.ppm", ticks / 20);
        save_ppm(bitmap, DIM, DIM, filename);
    }
    
    printf("Generated 5 ripple effect frames!\n");
    
    // Cleanup
    free(bitmap);
    HANDLE_ERROR(cudaFree(dev_bitmap));
    
    return 0;
}