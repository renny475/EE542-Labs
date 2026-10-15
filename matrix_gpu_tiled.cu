// EE542 Lab 6 - Naive vs shared-memory tiled CUDA matrix multiplication
// Build: nvcc -O2 -arch=sm_75 matrix_gpu_tiled.cu -o matrix_gpu_tiled
// Run:   ./matrix_gpu_tiled            (N = 512, 1024, 2048; writes results_gpu.csv)
//        ./matrix_gpu_tiled 256 4096   (any list of N)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TRIALS 5

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- Naive kernel (handout, Part 2) ----
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---- Tiled kernel (handout, verbatim) ----
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Check 100 random elements of C against a double-precision CPU reference
static int check(const float *A, const float *B, const float *C, int N, double *max_rel) {
    int bad = 0;
    *max_rel = 0.0;
    srand(123);
    for (int s = 0; s < 100; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++)
            ref += (double)A[(size_t)i * N + k] * B[(size_t)k * N + j];
        double rel = fabs(C[(size_t)i * N + j] - ref) / (fabs(ref) + 1e-12);
        if (rel > *max_rel) *max_rel = rel;
        if (rel > 1e-3) bad++;
    }
    return bad;
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

// Launch `which` kernel TRIALS times (after one warm-up); return median ms
static float time_kernel(int which, float *dA, float *dB, float *dC, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    float ms[TRIALS];
    for (int t = -1; t < TRIALS; t++) {          // t = -1 is the warm-up
        CUDA_CHECK(cudaEventRecord(s));
        if (which == 0) matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
        else            matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        if (t >= 0) CUDA_CHECK(cudaEventElapsedTime(&ms[t], s, e));
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    qsort(ms, TRIALS, sizeof(float), cmp_float);
    return ms[TRIALS / 2];
}

// Run naive and tiled kernels for one N; append results to csv. Returns 1 if both pass.
static int run_one(int N, FILE *csv);

int main(int argc, char **argv) {
    // No arguments: the handout sizes. Arguments: any list of N, e.g. ./matrix_gpu_tiled 256 4096
    int default_Ns[] = {512, 1024, 2048};
    int count = (argc > 1) ? argc - 1 : 3;

    FILE *csv = fopen("results_gpu.csv", "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "N,kernel,median_ms,gflops,correct\n");

    int all_ok = 1;
    for (int i = 0; i < count; i++) {
        int N = (argc > 1) ? atoi(argv[i + 1]) : default_Ns[i];
        if (N <= 0 || N > 46340) { fprintf(stderr, "invalid N: %d\n", N); return 1; }
        if (!run_one(N, csv)) all_ok = 0;
        fflush(csv);
        printf("\n");
    }
    fclose(csv);
    printf("Saved results_gpu.csv\n");
    return all_ok ? 0 : 1;
}

static int run_one(int N, FILE *csv) {
    size_t n2 = (size_t)N * N, size = n2 * sizeof(float);

    float *hA = (float *)malloc(size), *hB = (float *)malloc(size), *hC = (float *)malloc(size);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); exit(1); }
    srand(42);
    for (size_t i = 0; i < n2; i++) {
        hA[i] = rand() % 100 / 100.0f;
        hB[i] = rand() % 100 / 100.0f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));
    CUDA_CHECK(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

    const char *names[2] = {"naive", "tiled"};
    float med[2];
    int all_ok = 1;
    for (int w = 0; w < 2; w++) {
        CUDA_CHECK(cudaMemset(dC, 0, size));     // clear so a broken kernel can't pass
        med[w] = time_kernel(w, dA, dB, dC, N);
        CUDA_CHECK(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));
        double max_rel;
        int bad = check(hA, hB, hC, N, &max_rel);
        if (bad) all_ok = 0;
        double gflops = 2.0 * N * N * N / (med[w] / 1000.0) / 1e9;
        printf("%s kernel (N=%d): %.3f ms median of %d (%.2f GFLOPS) | Correctness: %s (max rel err %.2e)\n",
               names[w], N, med[w], TRIALS, gflops, bad == 0 ? "PASS" : "FAIL", max_rel);
        fprintf(csv, "%d,%s,%.4f,%.2f,%s\n", N, names[w], med[w], gflops, bad == 0 ? "PASS" : "FAIL");
    }
    printf("Tiled speedup over naive (N=%d): %.2fx\n", N, med[0] / med[1]);

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return all_ok;
}
