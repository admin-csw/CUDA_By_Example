#include <stdio.h>
#include <stdlib.h>
#include "cuda_common.h"

#define DIM 1000

struct cuComplex {
    float r;
    float i;
    
    __device__ cuComplex( float a, float b ) : r(a), i(b) {}
    
    __device__ float magnitude2( void ) { return r*r + i*i; }

    __device__ cuComplex operator*( const cuComplex& a ) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    
    __device__ cuComplex operator+( const cuComplex& a ) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);

    int i=0;
    for ( i=0; i<200; i++ ) {
        a = a * a + c;
        if ( a.magnitude2() > 1000 ) return 0;
    }

    return 1;
}

__global__ void kernel( unsigned char *ptr ) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    if (x < DIM && y < DIM) {
        // now calculate the value at that position
        int juliaValue = julia( x, y );
        ptr[offset*4+0] = 255 * juliaValue;  // R
        ptr[offset*4+1] = 0;                 // G
        ptr[offset*4+2] = 0;                 // B
        ptr[offset*4+3] = 255;               // A
    }
}

void save_ppm(const char* filename, unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }
    
    // PPM 헤더 작성
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    
    // 픽셀 데이터 작성 (RGBA -> RGB 변환)
    for (int i = 0; i < width * height; i++) {
        fputc(data[i*4+0], fp);  // R
        fputc(data[i*4+1], fp);  // G  
        fputc(data[i*4+2], fp);  // B
        // A 채널은 무시
    }
    
    fclose(fp);
    printf("GPU Julia Set saved as %s (%dx%d pixels)\n", filename, width, height);
}

int main(void) {
    // 호스트 메모리 할당
    unsigned char* host_bitmap = new unsigned char[DIM * DIM * 4];
    unsigned char* dev_bitmap;

    // GPU 메모리 할당
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, DIM * DIM * 4 ) );

    printf("Generating GPU Julia Set (%dx%d)...\n", DIM, DIM);

    // GPU 커널 실행 (1000x1000 블록, 각 블록당 1 스레드)
    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>( dev_bitmap );

    // GPU에서 호스트로 데이터 복사
    HANDLE_ERROR( cudaMemcpy( host_bitmap, dev_bitmap,
                            DIM * DIM * 4,
                            cudaMemcpyDeviceToHost ) );

    // PPM 이미지 파일로 저장
    save_ppm("gpu_julia_set.ppm", host_bitmap, DIM, DIM);

    // 메모리 해제
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    delete[] host_bitmap;

    printf("Complete! GPU acceleration used for parallel computation.\n");
    return 0;
}