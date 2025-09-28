#include <stdio.h>

#define N 10

void add(int *a, int *b, int *c) {
    int tid = 0;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid++;
    }
}

int main() {
    int a[N], b[N], c[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Perform vector addition
    add(a, b, c);

    // Print the result
    printf("Resultant vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    return 0;
}