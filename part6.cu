// EE542 Lab 6 - Part 6.1: CPU vs naive CUDA vs tiled CUDA vs cuBLAS (same machine)
// Build: nvcc -O2 -arch=sm_75 part6.cu -o part6 -lcublas
// Run:   ./part6                 (N = 512, 1024, 2048, 4096; writes results_part6.csv)
//        ./part6 256 512 8192    (any list of N)
// The CPU version is skipped above N = 2048 (it would take many minutes);
// for those sizes correctness is checked on 100 sampled elements instead.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16
#define GPU_TRIALS 5
#define CPU_MAX_N 2048

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n",             \
                    __FILE__, __LINE__, (int)st);                          \
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

// ---- cuBLAS (handout, Step 6.1) ----
// cuBLAS assumes column-major storage; our matrices are row-major.
// A row-major matrix read as column-major is its transpose, so passing (B, A)
// makes cuBLAS compute B^T * A^T = (A * B)^T in column-major, which is exactly
// C = A * B in row-major. No explicit transpose is needed.
static void gemm_cublas(cublasHandle_t h, const float *dA, const float *dB, float *dC, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             dB, N,ㄎ
                             dA, N,
                             &beta,
                             dC, N));
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

// Full check against the CPU result (all elements)
static double err_full(const float *ref, const float *got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-12);
        if (r > m) m = r;
    }
    return m;
}

// Sampled check (100 elements, double-precision reference) when the CPU run is skipped
static double err_sampled(const float *A, const float *B, const float *got, int N) {
    double m = 0.0;
    srand(123);
    for (int s = 0; s < 100; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++)
            ref += (double)A[(size_t)i * N + k] * B[(size_t)k * N + j];
        double r = fabs(got[(size_t)i * N + j] - ref) / (fabs(ref) + 1e-12);
        if (r > m) m = r;
    }
    return m;
}

// Median GPU time in ms over GPU_TRIALS runs (after one warm-up).
// which: 0 = naive, 1 = tiled, 2 = cuBLAS
static double time_gpu(int which, cublasHandle_t h, float *dA, float *dB, float *dC, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    double ms[GPU_TRIALS];
    for (int t = -1; t < GPU_TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(s));
        if (which == 0)      matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
        else if (which == 1) matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
        else                 gemm_cublas(h, dA, dB, dC, N);
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
    int default_Ns[] = {512, 1024, 2048, 4096};
    int count = (argc > 1) ? argc - 1 : 4;
    const char *names[3] = {"Naive CUDA", "Tiled CUDA", "cuBLAS    "};

    // One-time costs: CUDA context, then cuBLAS handle
    double t0 = now_sec();
    CUDA_CHECK(cudaFree(0));
    double ctx_ms = (now_sec() - t0) * 1000.0;
    cublasHandle_t handle;
    t0 = now_sec();
    CUBLAS_CHECK(cublasCreate(&handle));
    double handle_ms = (now_sec() - t0) * 1000.0;
    printf("CUDA context initialization (one-time): %.1f ms\n", ctx_ms);
    printf("cuBLAS handle creation (one-time):      %.1f ms\n\n", handle_ms);

    FILE *csv = fopen("results_part6.csv", "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "N,cpu_s,naive_ms,tiled_ms,cublas_ms,h2d_ms,d2h_ms,"
                 "naive_err,tiled_err,cublas_err,check\n");

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

        // ---- CPU ----
        int run_cpu = (N <= CPU_MAX_N);
        double cpu_s = -1.0;
        if (run_cpu) {
            printf("N = %d: running CPU version (may take a while)...\n", N);
            fflush(stdout);
            int trials = (N <= 1024) ? 3 : 1;
            double ct[3];
            for (int t = 0; t < trials; t++) {
                double s = now_sec();
                matrixMultiplyCPU(hA, hB, hRef, N);
                ct[t] = now_sec() - s;
            }
            qsort(ct, trials, sizeof(double), cmp_double);
            cpu_s = ct[trials / 2];
        }

        // ---- GPU ----
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, size));
        CUDA_CHECK(cudaMalloc(&dB, size));
        CUDA_CHECK(cudaMalloc(&dC, size));
        double h2d = time_copy(dA, hA, size, cudaMemcpyHostToDevice)
                   + time_copy(dB, hB, size, cudaMemcpyHostToDevice);

        double ms[3], err[3], d2h = 0.0;
        for (int w = 0; w < 3; w++) {
            CUDA_CHECK(cudaMemset(dC, 0, size));
            ms[w] = time_gpu(w, handle, dA, dB, dC, N);
            double c = time_copy(hC, dC, size, cudaMemcpyDeviceToHost);
            if (w == 0) d2h = c;
            err[w] = run_cpu ? err_full(hRef, hC, n2) : err_sampled(hA, hB, hC, N);
            if (err[w] > 1e-3) all_ok = 0;
        }
        double copies = h2d + d2h;

        printf("N = %d\n", N);
        if (run_cpu) printf("  CPU (C)    : %10.4f s\n", cpu_s);
        else         printf("  CPU (C)    :    skipped (N > %d)\n", CPU_MAX_N);
        for (int w = 0; w < 3; w++) {
            double gflops = 2.0 * N * N * N / (ms[w] / 1000.0) / 1e9;
            printf("  %s : %9.3f ms  %8.1f GFLOPS  end-to-end %9.3f ms  %s (max rel err %.2e)",
                   names[w], ms[w], gflops, ms[w] + copies,
                   err[w] <= 1e-3 ? "PASS" : "FAIL", err[w]);
            if (run_cpu)
                printf("  speedup %.1fx kernel / %.1fx e2e",
                       cpu_s * 1000.0 / ms[w], cpu_s * 1000.0 / (ms[w] + copies));
            printf("\n");
        }
        printf("  Copies     : H2D %.3f ms + D2H %.3f ms = %.3f ms\n", h2d, d2h, copies);
        printf("  cuBLAS vs tiled: %.2fx faster\n\n", ms[1] / ms[2]);

        fprintf(csv, "%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.3e,%.3e,%s\n",
                N, cpu_s, ms[0], ms[1], ms[2], h2d, d2h, err[0], err[1], err[2],
                run_cpu ? "full" : "sampled");
        fflush(csv);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hRef); free(hC);
    }
    fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));
    printf("Saved results_part6.csv\n");
    return all_ok ? 0 : 1;
}
