#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_common.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f
#define FRAMES \
    200  // Number of animation frames (extended for longer simulation)

// globals needed by the update routine
struct DataBlock {
    unsigned char* output_bitmap;
    float* dev_inSrc;
    float* dev_outSrc;
    float* dev_constSrc;
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__device__ unsigned char value(float n1, float n2, int hue) {
    if (hue > 360)
        hue -= 360;
    else if (hue < 0)
        hue += 360;

    if (hue < 60) return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
    if (hue < 180) return (unsigned char)(255 * n2);
    if (hue < 240)
        return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
    return (unsigned char)(255 * n1);
}

__global__ void float_to_color(unsigned char* optr, const float* outSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
    float m1, m2;

    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    optr[offset * 4 + 0] = value(m1, m2, h + 120);
    optr[offset * 4 + 1] = value(m1, m2, h);
    optr[offset * 4 + 2] = value(m1, m2, h - 120);
    optr[offset * 4 + 3] = 255;
}

__global__ void copy_const_kernel(float* iptr, const float* cptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float* outSrc, const float* inSrc) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0) left++;
    if (x == DIM - 1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0) top += DIM;
    if (y == DIM - 1) bottom -= DIM;

    outSrc[offset] =
        inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] +
                                 inSrc[right] - 4 * inSrc[offset]);
}

void save_ppm(unsigned char* bitmap, int width, int height,
              const char* filename) {
    FILE* file = fopen(filename, "wb");
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
}

void animate_heat(DataBlock* d) {
    HANDLE_ERROR(cudaEventRecord(d->start, 0));

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    int image_size = DIM * DIM * 4;
    unsigned char* host_bitmap = (unsigned char*)malloc(image_size);

    printf("Generating Heat Animation (%d frames)...\n", FRAMES);

    for (int frame = 0; frame < FRAMES; frame++) {
        // Run simulation steps for this frame (90 iterations per frame for
        // longer sequence)
        for (int i = 0; i < 90; i++) {
            copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc,
                                                   d->dev_constSrc);
            blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
            swap(d->dev_inSrc, d->dev_outSrc);
        }

        // Convert to color and save frame
        float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
        HANDLE_ERROR(cudaMemcpy(host_bitmap, d->output_bitmap, image_size,
                                cudaMemcpyDeviceToHost));

        // Save frame
        char filename[64];
        sprintf(filename, "heat_frame_%03d.ppm", frame);
        save_ppm(host_bitmap, DIM, DIM, filename);

        if (frame % 20 == 0) {
            printf("Frame %d/%d saved (%.1f%% complete)\n", frame, FRAMES,
                   (frame * 100.0f) / FRAMES);
        }
    }

    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    printf("\nAnimation Generation Complete!\n");
    printf("Total time: %3.1f ms\n", elapsedTime);
    printf("Time per frame: %3.1f ms\n", elapsedTime / FRAMES);
    printf("Generated %d frames: heat_frame_000.ppm to heat_frame_%03d.ppm\n",
           FRAMES, FRAMES - 1);

    free(host_bitmap);
}

void cleanup(DataBlock* d) {
    HANDLE_ERROR(cudaFree(d->dev_inSrc));
    HANDLE_ERROR(cudaFree(d->dev_outSrc));
    HANDLE_ERROR(cudaFree(d->dev_constSrc));
    HANDLE_ERROR(cudaFree(d->output_bitmap));
    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
    DataBlock data;
    data.totalTime = 0;
    data.frames = 0;

    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    int image_size = DIM * DIM * 4;  // RGBA

    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, image_size));
    HANDLE_ERROR(
        cudaMalloc((void**)&data.dev_inSrc, DIM * DIM * sizeof(float)));
    HANDLE_ERROR(
        cudaMalloc((void**)&data.dev_outSrc, DIM * DIM * sizeof(float)));
    HANDLE_ERROR(
        cudaMalloc((void**)&data.dev_constSrc, DIM * DIM * sizeof(float)));

    float* temp = (float*)malloc(DIM * DIM * sizeof(float));

    // Initialize temperature field
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;

        int x = i % DIM;
        int y = i / DIM;

        // Hot region in the center
        if ((x > 300) && (x < 600) && (y > 300) && (y < 600))
            temp[i] = MAX_TEMP;
    }

    // Add some heat sources
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;

    // Hot region at bottom
    for (int y = 800; y < 900; y++)
        for (int x = 400; x < 500; x++) temp[y * DIM + x] = MIN_TEMP;

    // Copy constant heat sources to GPU
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, DIM * DIM * sizeof(float),
                            cudaMemcpyHostToDevice));

    // Hot region at bottom left
    for (int y = 800; y < DIM; y++)
        for (int x = 0; x < 200; x++) temp[y * DIM + x] = MAX_TEMP;

    // Copy initial temperature field to GPU
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, DIM * DIM * sizeof(float),
                            cudaMemcpyHostToDevice));

    free(temp);

    printf("CUDA Heat Animation (Chapter 7)\n");
    printf("Grid Size: %dx%d\n", DIM, DIM);
    printf("Animation Frames: %d\n", FRAMES);
    printf("Speed: %f\n", SPEED);
    printf("Max Temp: %f\n", MAX_TEMP);
    printf("Min Temp: %f\n", MIN_TEMP);
    printf("\n");

    // Generate animation frames
    animate_heat(&data);

    printf("\nTo create GIF animation, use:\n");
    printf("convert heat_frame_*.ppm -delay 10 heat_animation.gif\n");

    cleanup(&data);

    return 0;
}