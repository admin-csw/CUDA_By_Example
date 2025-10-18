#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_common.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f
#define FRAMES 100  // Number of animation frames

// 2D texture declarations (texture<float, 2> means 2D)
texture<float, 2> textConstSrc;
texture<float, 2> textIn;
texture<float, 2> textOut;

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

__global__ void copy_const_kernel(float* iptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // Use tex2D for 2D texture access
    float c = tex2D(textConstSrc, x, y);
    if (c != 0) iptr[offset] = c;
}

__global__ void blend_kernel(float* dst, bool dstOut) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;

    if (dstOut) {
        t = tex2D(textIn, x, y - 1);
        l = tex2D(textIn, x - 1, y);
        c = tex2D(textIn, x, y);
        r = tex2D(textIn, x + 1, y);
        b = tex2D(textIn, x, y + 1);
    } else {
        t = tex2D(textOut, x, y - 1);
        l = tex2D(textOut, x - 1, y);
        c = tex2D(textOut, x, y);
        r = tex2D(textOut, x + 1, y);
        b = tex2D(textOut, x, y + 1);
    }

    dst[offset] = c + SPEED * (t + b + l + r - 4 * c);
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

void animate_heat_2d_texture(DataBlock* d) {
    HANDLE_ERROR(cudaEventRecord(d->start, 0));

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    int image_size = DIM * DIM * 4;
    unsigned char* host_bitmap = (unsigned char*)malloc(image_size);

    printf(
        "Generating Heat Animation with 2D Texture Memory (%d frames, 100x "
        "speed)...\n",
        FRAMES);

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;

    for (int frame = 0; frame < FRAMES; frame++) {
        // Run simulation steps for this frame (90 iterations per frame - 100x
        // faster)
        for (int i = 0; i < 90; i++) {
            float* in;
            float* out;
            if (dstOut) {
                in = d->dev_inSrc;
                out = d->dev_outSrc;
            } else {
                in = d->dev_outSrc;
                out = d->dev_inSrc;
            }
            copy_const_kernel<<<blocks, threads>>>(in);
            blend_kernel<<<blocks, threads>>>(out, dstOut);
            dstOut = !dstOut;
        }

        // Convert to color and save frame
        float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
        HANDLE_ERROR(cudaMemcpy(host_bitmap, d->output_bitmap, image_size,
                                cudaMemcpyDeviceToHost));

        // Save frame
        char filename[64];
        sprintf(filename, "heat_2d_texture_fast_frame_%03d.ppm", frame);
        save_ppm(host_bitmap, DIM, DIM, filename);

        if (frame % 10 == 0) {
            printf("Frame %d/%d saved\n", frame, FRAMES);
        }
    }

    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    printf(
        "\n2D Texture Memory Fast Animation Generation Complete! (100x "
        "speed)\n");
    printf("Total time: %3.1f ms\n", elapsedTime);
    printf("Time per frame: %3.1f ms (1000 simulation steps per frame)\n",
           elapsedTime / FRAMES);
    printf(
        "Generated %d frames: heat_2d_texture_fast_frame_000.ppm to "
        "heat_2d_texture_fast_frame_%03d.ppm\n",
        FRAMES, FRAMES - 1);

    free(host_bitmap);
}

void cleanup(DataBlock* d) {
    cudaUnbindTexture(textIn);
    cudaUnbindTexture(textOut);
    cudaUnbindTexture(textConstSrc);
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

    int imageSize = DIM * DIM * sizeof(float);
    int bitmapSize = DIM * DIM * 4;  // RGBA

    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmapSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));

    // Setup 2D texture memory
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaBindTexture2D(NULL, textConstSrc, data.dev_constSrc, desc,
                                   DIM, DIM, sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL, textIn, data.dev_inSrc, desc, DIM, DIM,
                                   sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL, textOut, data.dev_outSrc, desc, DIM,
                                   DIM, sizeof(float) * DIM));

    // Initialize temperature field
    float* temp = (float*)malloc(imageSize);
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }

    // Add some heat/cold sources
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;

    // Cold region at bottom center
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }

    HANDLE_ERROR(
        cudaMemcpy(data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice));

    // Hot region at bottom left
    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    HANDLE_ERROR(
        cudaMemcpy(data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice));
    free(temp);

    printf("CUDA Heat Animation with 2D Texture Memory (Chapter 7)\n");
    printf("Grid Size: %dx%d\n", DIM, DIM);
    printf("Animation Frames: %d\n", FRAMES);
    printf("Speed: %f\n", SPEED);
    printf("Max Temp: %f\n", MAX_TEMP);
    printf("Min Temp: %f\n", MIN_TEMP);
    printf("Using GPU 2D Texture Memory with tex2D() calls\n");
    printf("\n");

    // Generate animation frames
    animate_heat_2d_texture(&data);

    printf("\nTo create GIF animation, use:\n");
    printf(
        "convert heat_2d_texture_fast_frame_*.ppm -delay 10 "
        "heat_2d_texture_fast_animation.gif\n");

    cleanup(&data);

    return 0;
}