#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplyCPU_ikj(const float* restrict A, const float* restrict B,
                           float* restrict C, int N) {
    for (size_t i = 0; i < (size_t)N * N; i++) C[i] = 0.0f;

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = N * N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    clock_t start = clock();
    matrixMultiplyCPU(A, B, C, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU execution time (N=%d): %f seconds\n", N, elapsed);

    free(A);
    free(B);
    free(C);


    return 0;
}