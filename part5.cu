// EE542 Lab 6 - Part 5: CPU vs naive CUDA vs tiled CUDA on the same machine
// Build: nvcc -O2 -arch=sm_75 part5.cu -o part5
// Run:   ./part5                 (N = 512, 1024, 2048; writes results_part5.csv)
//        ./part5 256 512 4096    (any list of N)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define GPU_TRIALS 5

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- CPU (handout, Part 1) ----
void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---- Naive CUDA (handout, Part 2) ----
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

// ---- Optimized CUDA: shared-memory tiling (handout, verbatim) ----
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

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Max relative error of `got` against the CPU result over all elements
static double max_rel_err(const float *ref, const float *got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-12);
        if (r > m) m = r;
    }
    return m;
}

// Median kernel time in ms over GPU_TRIALS launches (after one warm-up)
static double time_kernel(int which, float *dA, float *dB, float *dC, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    double ms[GPU_TRIALS];
    for (int t = -1; t < GPU_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(s));
        if (which == 0) matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
        else            matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float f;
        CUDA_CHECK(cudaEventElapsedTime(&f, s, e));
        if (t >= 0) ms[t] = f;
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    qsort(ms, GPU_TRIALS, sizeof(double), cmp_double);
    return ms[GPU_TRIALS / 2];
}

// Time one cudaMemcpy in ms with CUDA events
static double time_copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float f;
    CUDA_CHECK(cudaEventElapsedTime(&f, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return f;
}

int main(int argc, char **argv) {
    int default_Ns[] = {512, 1024, 2048};
    int count = (argc > 1) ? argc - 1 : 3;

    // One-time CUDA context creation: a fixed overhead paid by the first CUDA call
    double t0 = now_sec();
    CUDA_CHECK(cudaFree(0));
    double ctx_ms = (now_sec() - t0) * 1000.0;
    printf("CUDA context initialization (one-time): %.1f ms\n\n", ctx_ms);

    FILE *csv = fopen("results_part5.csv", "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "N,cpu_s,naive_kernel_ms,tiled_kernel_ms,h2d_ms,d2h_ms,"
                 "naive_total_ms,tiled_total_ms,naive_err,tiled_err,ctx_init_ms\n");

    int all_ok = 1;
    for (int idx = 0; idx < count; idx++) {
        int N = (argc > 1) ? atoi(argv[idx + 1]) : default_Ns[idx];
        if (N <= 0 || N > 46340) { fprintf(stderr, "invalid N: %d\n", N); return 1; }
        size_t n2 = (size_t)N * N, size = n2 * sizeof(float);

        float *hA = (float *)malloc(size), *hB = (float *)malloc(size);
        float *hRef = (float *)malloc(size), *hC = (float *)malloc(size);
        if (!hA || !hB || !hRef || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
        srand(42);
        for (size_t i = 0; i < n2; i++) {
            hA[i] = rand() % 100 / 100.0f;
            hB[i] = rand() % 100 / 100.0f;
        }

        // ---- CPU: median of 3 for N <= 1024, single run above (it takes minutes) ----
        int cpu_trials = (N <= 1024) ? 3 : 1;
        double cpu_t[3];
        for (int t = 0; t < cpu_trials; t++) {
            double s = now_sec();
            matrixMultiplyCPU(hA, hB, hRef, N);
            cpu_t[t] = now_sec() - s;
        }
        qsort(cpu_t, cpu_trials, sizeof(double), cmp_double);
        double cpu_s = cpu_t[cpu_trials / 2];

        // ---- GPU ----
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, size));
        CUDA_CHECK(cudaMalloc(&dB, size));
        CUDA_CHECK(cudaMalloc(&dC, size));

        double h2d = time_copy(dA, hA, size, cudaMemcpyHostToDevice)
                   + time_copy(dB, hB, size, cudaMemcpyHostToDevice);

        double kms[2], err[2], d2h = 0.0;
        for (int w = 0; w < 2; w++) {
            CUDA_CHECK(cudaMemset(dC, 0, size));
            kms[w] = time_kernel(w, dA, dB, dC, N);
            double c = time_copy(hC, dC, size, cudaMemcpyDeviceToHost);
            if (w == 0) d2h = c;
            err[w] = max_rel_err(hRef, hC, n2);
            if (err[w] > 1e-3) all_ok = 0;
        }
        double tot[2] = {h2d + kms[0] + d2h, h2d + kms[1] + d2h};

        printf("N = %d\n", N);
        printf("  CPU (C)          : %10.4f s\n", cpu_s);
        printf("  Naive CUDA kernel: %10.3f ms   end-to-end %10.3f ms   %s (max rel err %.2e)\n",
               kms[0], tot[0], err[0] <= 1e-3 ? "PASS" : "FAIL", err[0]);
        printf("  Tiled CUDA kernel: %10.3f ms   end-to-end %10.3f ms   %s (max rel err %.2e)\n",
               kms[1], tot[1], err[1] <= 1e-3 ? "PASS" : "FAIL", err[1]);
        printf("  Copies           : H2D %.3f ms + D2H %.3f ms = %.3f ms\n", h2d, d2h, h2d + d2h);
        printf("  Speedup vs CPU   : naive %.1fx kernel / %.1fx end-to-end;"
               " tiled %.1fx kernel / %.1fx end-to-end\n\n",
               cpu_s * 1000.0 / kms[0], cpu_s * 1000.0 / tot[0],
               cpu_s * 1000.0 / kms[1], cpu_s * 1000.0 / tot[1]);

        fprintf(csv, "%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.3e,%.1f\n",
                N, cpu_s, kms[0], kms[1], h2d, d2h, tot[0], tot[1], err[0], err[1], ctx_ms);
        fflush(csv);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hRef); free(hC);
    }
    fclose(csv);
    printf("Saved results_part5.csv\n");
    return all_ok ? 0 : 1;
}
