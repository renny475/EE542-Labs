// EE542 Lab 6 - Optimizations from the CUDA C++ Best Practices Guide, one change at a time
//   V1 Baseline : malloc         + cudaMemcpy                  + tiled kernel (handout)
//   V2 Pinned   : cudaMallocHost + cudaMemcpy                  + tiled kernel       (Guide 10.1.1)
//   V3 Streams  : cudaMallocHost + cudaMemcpyAsync, 4 streams  + tiled kernel       (Guide 10.1.2)
//   V4 Register : cudaMallocHost + cudaMemcpy                  + register-tiled kernel,
//                 1, 2, 4 (and 16) outputs per thread                               (Guide 11.4)
// cuBLAS is run only as the correctness reference and a performance ceiling.
//
// Build: nvcc -O2 -arch=sm_75 part7.cu -o part7 -lcublas
// Run:   ./part7                 (N = 512, 1024, 2048, 4096; writes results_part7.csv)
//        ./part7 1024 8192       (any list of N; each must be a multiple of 64)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16
#define NSTREAMS 4
#define TRIALS 5

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                             \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t cb_status_ = (call);                                \
        if (cb_status_ != CUBLAS_STATUS_SUCCESS) {                         \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n",             \
                    __FILE__, __LINE__, (int)cb_status_);                  \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- Handout tiled kernel, unchanged (used by V1, V2, V3) ----
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

// ---- V4: register-tiled kernel, each thread computes a TM x TN block of outputs ----
// Block = 16 x 16 threads; it computes a (16*TM) x (16*TN) tile of C, stepping through
// K in slices of 16. Per K-step a thread reads TM + TN values from shared memory and does
// TM * TN multiply-adds, all accumulated in registers:
//   1x1 -> 2 shared loads per FMA, 2x1 -> 1.5, 2x2 -> 1.0, 4x4 -> 0.5.
// Thread (tx, ty) owns rows ty + 16*i and cols tx + 16*j, so within a warp the shared-memory
// reads are either broadcasts (As) or consecutive words (Bs): no bank conflicts.
template <int TM, int TN>
__global__ void matrixMultiplyRegTiled(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C, int N) {
    const int BMr = 16 * TM, BNc = 16 * TN, BK = 16;
    __shared__ float As[16 * TM][16];
    __shared__ float Bs[16][16 * TN];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;
    int rowBase = blockIdx.y * BMr;
    int colBase = blockIdx.x * BNc;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // A tile: (16*TM) x 16 values = 256*TM, i.e. TM loads per thread
#pragma unroll
        for (int l = 0; l < TM; l++) {
            int e = tid + 256 * l;
            int ar = e / BK, ac = e % BK;
            int gr = rowBase + ar, gc = k0 + ac;
            As[ar][ac] = (gr < N && gc < N) ? A[(size_t)gr * N + gc] : 0.0f;
        }
        // B tile: 16 x (16*TN) values = 256*TN, i.e. TN loads per thread
#pragma unroll
        for (int l = 0; l < TN; l++) {
            int e = tid + 256 * l;
            int br = e / BNc, bc = e % BNc;
            int gr = k0 + br, gc = colBase + bc;
            Bs[br][bc] = (gr < N && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty + 16 * i][k];
#pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = rowBase + ty + 16 * i;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = colBase + tx + 16 * j;
            if (r < N && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------------------
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_double);
    return v[n / 2];
}
static double max_rel_err(const float *ref, const float *got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = fabs((double)got[i] - ref[i]) / (fabs((double)ref[i]) + 1e-12);
        if (r > m) m = r;
    }
    return m;
}

// Kernel variants: 0 = handout tiled, 1 = 1x1, 2 = 2x1, 3 = 2x2, 4 = 4x4 outputs per thread
#define NVARIANTS 5
static const char *VNAME[NVARIANTS] = {"handout tiled (1 output)", "register 1x1 (1 output)",
                                       "register 2x1 (2 outputs)", "register 2x2 (4 outputs)",
                                       "register 4x4 (16 outputs)"};
static const char *VSHAPE[NVARIANTS] = {"tiled", "1x1", "2x1", "2x2", "4x4"};
static const int VOUT[NVARIANTS] = {1, 1, 2, 4, 16};

// Launch kernel variant v on rows [0, rows) of a row-major problem with N columns
// (A and C may already be offset to a row block)
static void launch(int v, float *A, float *B, float *C, int N, int rows, cudaStream_t s) {
    dim3 blk(16, 16);
    switch (v) {
    case 0: { dim3 g((N + 15) / 16, (rows + 15) / 16);
              matrixMultiplyTiled<<<g, blk, 0, s>>>(A, B, C, N); break; }
    case 1: { dim3 g((N + 15) / 16, (rows + 15) / 16);
              matrixMultiplyRegTiled<1, 1><<<g, blk, 0, s>>>(A, B, C, N); break; }
    case 2: { dim3 g((N + 15) / 16, (rows + 31) / 32);
              matrixMultiplyRegTiled<2, 1><<<g, blk, 0, s>>>(A, B, C, N); break; }
    case 3: { dim3 g((N + 31) / 32, (rows + 31) / 32);
              matrixMultiplyRegTiled<2, 2><<<g, blk, 0, s>>>(A, B, C, N); break; }
    case 4: { dim3 g((N + 63) / 64, (rows + 63) / 64);
              matrixMultiplyRegTiled<4, 4><<<g, blk, 0, s>>>(A, B, C, N); break; }
    }
    CUDA_CHECK(cudaGetLastError());
}

// Median kernel-only time (ms), CUDA events
static double time_kernel(int v, float *dA, float *dB, float *dC, int N) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    double ms[TRIALS];
    for (int t = -1; t < TRIALS; t++) {
        CUDA_CHECK(cudaEventRecord(s));
        launch(v, dA, dB, dC, N, N, 0);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float f;
        CUDA_CHECK(cudaEventElapsedTime(&f, s, e));
        if (t >= 0) ms[t] = f;
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return median(ms, TRIALS);
}

// Synchronous end-to-end run (V1, V2, V4): copy A, B in; kernel; copy C out.
// Returns median total ms (wall clock); medians of H2D / kernel / D2H ms (CUDA events).
static double run_sync(int v, const float *hA, const float *hB, float *hC,
                       float *dA, float *dB, float *dC, int N,
                       double *h2d, double *ker, double *d2h) {
    size_t size = (size_t)N * N * sizeof(float);
    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
    double tot[TRIALS], a[TRIALS], b[TRIALS], c[TRIALS];
    for (int t = -1; t < TRIALS; t++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double s = now_sec();
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        launch(v, dA, dB, dC, N, N, 0);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        double wall = (now_sec() - s) * 1000.0;
        float f1, f2, f3;
        CUDA_CHECK(cudaEventElapsedTime(&f1, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&f2, e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&f3, e2, e3));
        if (t >= 0) { tot[t] = wall; a[t] = f1; b[t] = f2; c[t] = f3; }
    }
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2)); CUDA_CHECK(cudaEventDestroy(e3));
    *h2d = median(a, TRIALS); *ker = median(b, TRIALS); *d2h = median(c, TRIALS);
    return median(tot, TRIALS);
}

// V3: pinned memory + cudaMemcpyAsync + NSTREAMS streams, handout tiled kernel.
// B is needed by every chunk, so it is copied first; A and C are split into NSTREAMS
// row blocks, and each block's copy-in, kernel and copy-out run on its own stream so
// one block's transfers can overlap another block's computation.
static double run_streams(const float *hA, const float *hB, float *hC,
                          float *dA, float *dB, float *dC, int N, cudaStream_t *st) {
    size_t size = (size_t)N * N * sizeof(float);
    int rows = N / NSTREAMS;
    size_t chunk = (size_t)rows * N;
    cudaEvent_t bReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&bReady, cudaEventDisableTiming));
    double tot[TRIALS];
    for (int t = -1; t < TRIALS; t++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        double s = now_sec();
        CUDA_CHECK(cudaMemcpyAsync(dB, hB, size, cudaMemcpyHostToDevice, st[0]));
        CUDA_CHECK(cudaEventRecord(bReady, st[0]));
        for (int i = 0; i < NSTREAMS; i++) {
            size_t off = i * chunk;
            CUDA_CHECK(cudaMemcpyAsync(dA + off, hA + off, chunk * sizeof(float),
                                       cudaMemcpyHostToDevice, st[i]));
            CUDA_CHECK(cudaStreamWaitEvent(st[i], bReady, 0));
            launch(0, dA + off, dB, dC + off, N, rows, st[i]);
            CUDA_CHECK(cudaMemcpyAsync(hC + off, dC + off, chunk * sizeof(float),
                                       cudaMemcpyDeviceToHost, st[i]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double wall = (now_sec() - s) * 1000.0;
        if (t >= 0) tot[t] = wall;
    }
    CUDA_CHECK(cudaEventDestroy(bReady));
    return median(tot, TRIALS);
}

#define ROW_FMT "%-32s %9.3f %10.3f %9.3f %10.3f  %s\n"

int main(int argc, char **argv) {
    int default_Ns[] = {512, 1024, 2048, 4096};
    int count = (argc > 1) ? argc - 1 : 4;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaStream_t st[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) CUDA_CHECK(cudaStreamCreate(&st[i]));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | copy engines (asyncEngineCount): %d | streams used: %d\n\n",
           prop.name, prop.asyncEngineCount, NSTREAMS);

    FILE *csv = fopen("results_part7.csv", "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "N,version,outputs_per_thread,h2d_ms,kernel_ms,d2h_ms,total_ms,gflops_kernel,max_rel_err\n");

    int all_ok = 1;
    for (int idx = 0; idx < count; idx++) {
        int N = (argc > 1) ? atoi(argv[idx + 1]) : default_Ns[idx];
        if (N <= 0 || N > 46340 || N % 64) {
            fprintf(stderr, "invalid N: %d (must be a positive multiple of 64)\n", N);
            return 1;
        }
        size_t n2 = (size_t)N * N, size = n2 * sizeof(float);
        double gf = 2.0 * N * N * (double)N / 1e9;

        float *pA = (float *)malloc(size), *pB = (float *)malloc(size), *pC = (float *)malloc(size);
        float *ref = (float *)malloc(size);
        if (!pA || !pB || !pC || !ref) { fprintf(stderr, "malloc failed\n"); return 1; }
        float *kA, *kB, *kC;
        CUDA_CHECK(cudaMallocHost(&kA, size));
        CUDA_CHECK(cudaMallocHost(&kB, size));
        CUDA_CHECK(cudaMallocHost(&kC, size));
        srand(42);
        for (size_t i = 0; i < n2; i++) {
            pA[i] = kA[i] = rand() % 100 / 100.0f;
            pB[i] = kB[i] = rand() % 100 / 100.0f;
        }
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, size));
        CUDA_CHECK(cudaMalloc(&dB, size));
        CUDA_CHECK(cudaMalloc(&dC, size));

        // Reference result and performance ceiling: cuBLAS (column-major trick, Section 8)
        CUDA_CHECK(cudaMemcpy(dA, pA, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, pB, size, cudaMemcpyHostToDevice));
        const float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t cs, ce;
        CUDA_CHECK(cudaEventCreate(&cs)); CUDA_CHECK(cudaEventCreate(&ce));
        double cb[TRIALS];
        for (int t = -1; t < TRIALS; t++) {
            CUDA_CHECK(cudaEventRecord(cs));
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, dB, N, dA, N, &beta, dC, N));
            CUDA_CHECK(cudaEventRecord(ce));
            CUDA_CHECK(cudaEventSynchronize(ce));
            float f;
            CUDA_CHECK(cudaEventElapsedTime(&f, cs, ce));
            if (t >= 0) cb[t] = f;
        }
        CUDA_CHECK(cudaEventDestroy(cs)); CUDA_CHECK(cudaEventDestroy(ce));
        double cublas_ms = median(cb, TRIALS);
        CUDA_CHECK(cudaMemcpy(ref, dC, size, cudaMemcpyDeviceToHost));

        printf("==================== N = %d ====================\n", N);
        printf("%-32s %9s %10s %9s %10s  %s\n", "Version (end-to-end)", "H2D ms", "kernel ms", "D2H ms", "total ms", "check");

        double h2d, ker, d2h, tot, err;

        // ---- V1: baseline (pageable malloc) ----
        CUDA_CHECK(cudaMemset(dC, 0, size));
        tot = run_sync(0, pA, pB, pC, dA, dB, dC, N, &h2d, &ker, &d2h);
        err = max_rel_err(ref, pC, n2); if (err > 1e-3) all_ok = 0;
        double v1_tot = tot, v1_copy = h2d + d2h, v1_h2d = h2d, v1_d2h = d2h;
        printf(ROW_FMT, "V1 baseline (malloc)", h2d, ker, d2h, tot, err <= 1e-3 ? "PASS" : "FAIL");
        fprintf(csv, "%d,V1 baseline,1,%.4f,%.4f,%.4f,%.4f,%.1f,%.3e\n", N, h2d, ker, d2h, tot, gf / (ker / 1e3), err);

        // ---- V2: pinned memory ----
        CUDA_CHECK(cudaMemset(dC, 0, size));
        tot = run_sync(0, kA, kB, kC, dA, dB, dC, N, &h2d, &ker, &d2h);
        err = max_rel_err(ref, kC, n2); if (err > 1e-3) all_ok = 0;
        double v2_tot = tot, v2_copy = h2d + d2h, v2_ker = ker;
        printf(ROW_FMT, "V2 pinned (cudaMallocHost)", h2d, ker, d2h, tot, err <= 1e-3 ? "PASS" : "FAIL");
        fprintf(csv, "%d,V2 pinned,1,%.4f,%.4f,%.4f,%.4f,%.1f,%.3e\n", N, h2d, ker, d2h, tot, gf / (ker / 1e3), err);
        printf("   pinned bandwidth: H2D %.1f GB/s, D2H %.1f GB/s (pageable: %.1f / %.1f GB/s)\n",
               2.0 * size / 1e6 / h2d, size / 1e6 / d2h, 2.0 * size / 1e6 / v1_h2d, size / 1e6 / v1_d2h);

        // ---- V3: pinned + async copies + streams ----
        CUDA_CHECK(cudaMemset(dC, 0, size));
        memset(kC, 0, size);
        tot = run_streams(kA, kB, kC, dA, dB, dC, N, st);
        err = max_rel_err(ref, kC, n2); if (err > 1e-3) all_ok = 0;
        double v3_tot = tot;
        printf("%-32s %9s %10s %9s %10.3f  %s\n", "V3 pinned + 4 streams", "-", "-", "-", tot, err <= 1e-3 ? "PASS" : "FAIL");
        fprintf(csv, "%d,V3 streams,1,,,,%.4f,,%.3e\n", N, tot, err);

        // ---- V4: pinned + register-tiled kernels ----
        for (int v = 1; v < NVARIANTS; v++) {
            CUDA_CHECK(cudaMemset(dC, 0, size));
            memset(kC, 0, size);
            tot = run_sync(v, kA, kB, kC, dA, dB, dC, N, &h2d, &ker, &d2h);
            err = max_rel_err(ref, kC, n2); if (err > 1e-3) all_ok = 0;
            char label[64];
            snprintf(label, sizeof label, "V4 %s", VNAME[v]);
            printf(ROW_FMT, label, h2d, ker, d2h, tot, err <= 1e-3 ? "PASS" : "FAIL");
            fprintf(csv, "%d,V4 register %s,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.3e\n",
                    N, VSHAPE[v], VOUT[v], h2d, ker, d2h, tot, gf / (ker / 1e3), err);
        }

        // ---- Kernel-only comparison ----
        printf("\nKernel only (median of %d):\n", TRIALS);
        double k0 = 0;
        for (int v = 0; v < NVARIANTS; v++) {
            double k = time_kernel(v, dA, dB, dC, N);
            if (v == 0) k0 = k;
            printf("  %-27s %9.3f ms %7.0f GFLOPS  %5.2fx vs handout tiled  %4.0f%% of cuBLAS\n",
                   VNAME[v], k, gf / (k / 1e3), k0 / k, 100.0 * cublas_ms / k);
        }
        printf("  %-27s %9.3f ms %7.0f GFLOPS  (reference)\n", "cuBLAS", cublas_ms, gf / (cublas_ms / 1e3));
        fprintf(csv, "%d,cuBLAS reference,,,%.4f,,,%.1f,\n", N, cublas_ms, gf / (cublas_ms / 1e3));

        printf("\nSummary N=%d: pinned copies %.2fx faster than pageable | V2 total %.2fx faster than V1 | "
               "V3 total %.2fx faster than V2 (V2 copies %.3f ms + kernel %.3f ms)\n\n",
               N, v1_copy / v2_copy, v1_tot / v2_tot, v2_tot / v3_tot, v2_copy, v2_ker);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFreeHost(kA)); CUDA_CHECK(cudaFreeHost(kB)); CUDA_CHECK(cudaFreeHost(kC));
        free(pA); free(pB); free(pC); free(ref);
    }
    fclose(csv);
    for (int i = 0; i < NSTREAMS; i++) CUDA_CHECK(cudaStreamDestroy(st[i]));
    CUBLAS_CHECK(cublasDestroy(handle));
    printf("Saved results_part7.csv\n");
    return all_ok ? 0 : 1;
}
