#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

struct cuComplex {
    float r;
    float i;
    
    cuComplex( float a, float b ) : r(a), i(b) {}
    cuComplex() : r(0), i(0) {}
    
    float magnitude2( void ) { return r*r + i*i; }

    cuComplex operator*( const cuComplex& a ) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    
    cuComplex operator+( const cuComplex& a ) {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex z(jx, jy);

    int i=0;
    for ( i=0; i<200; i++ ) {
        z = z * z + c;
        if ( z.magnitude2() > 1000 ) return 0;
    }

    return 1;
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
    printf("Julia Set saved as %s (%dx%d pixels)\n", filename, width, height);
}

void kernel( unsigned char *ptr ) {
    int x, y;
    for ( y=0; y<DIM; y++ ) {
        for ( x=0; x<DIM; x++ ) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4+0] = 255 * juliaValue;  // R
            ptr[offset*4+1] = 0;                 // G
            ptr[offset*4+2] = 0;                 // B
            ptr[offset*4+3] = 255;               // A
        }
    }
}

int main(void) {
    // 메모리 할당
    unsigned char* pixels = new unsigned char[DIM * DIM * 4];
    
    printf("Generating Julia Set (%dx%d)...\n", DIM, DIM);
    
    // Julia Set 계산
    kernel(pixels);
    
    // PPM 이미지 파일로 저장
    save_ppm("julia_set.ppm", pixels, DIM, DIM);
    
    // 메모리 해제
    delete[] pixels;
    
    printf("Complete! You can view the image with: eog julia_set.ppm\n");
    return 0;
}