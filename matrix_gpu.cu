// EE542 Lab 6 - Naive CUDA matrix multiplication
// Build: nvcc -O2 -arch=sm_75 matrix_gpu.cu -o matrix_gpu
// Run:   ./matrix_gpu 1024
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Check every CUDA API call; print file, line and reason on failure
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Naive kernel from the handout: one thread computes one C[row][col]
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
#define TILE 16

// Share Memory Tiling 
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // 1. 每個 thread 搬一個 A 元素、一個 B 元素；超出邊界補 0
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();   // 2. 確保整個 tile 都載入完成

        // 3. 在 shared memory 上計算這個 tile 的部分乘積
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();   // 4. 確保大家都算完，才能進下一輪覆蓋 As/Bs
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0 || N > 46340) {
        fprintf(stderr, "invalid N\n");
        return 1;
    }
    size_t n2 = (size_t)N * N;
    size_t size = n2 * sizeof(float);

    // ---- Host memory ----
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    srand(42);
    for (size_t i = 0; i < n2; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // ---- Device memory ----
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    CUDA_CHECK(cudaEventCreate(&t3));

    // ---- Warm-up (CUDA context init; not timed) ----
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Timed run ----
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));

    float ms_h2d, ms_kernel, ms_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, t0, t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, t1, t2));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, t2, t3));
    double kernel_s = ms_kernel / 1000.0;
    double total_s = (ms_h2d + ms_kernel + ms_d2h) / 1000.0;

    // ---- Correctness: 100 random elements recomputed on CPU in double ----
    int bad = 0;
    double max_rel = 0.0;
    for (int s = 0; s < 100; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++) {
            ref += (double)h_A[(size_t)i * N + k] * h_B[(size_t)k * N + j];
        }
        double rel = fabs(h_C[(size_t)i * N + j] - ref) / (fabs(ref) + 1e-12);
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3) bad++;
    }

    printf("GPU kernel time (N=%d): %f seconds (%.2f GFLOPS)\n",
           N, kernel_s, 2.0 * N * N * N / kernel_s / 1e9);
    printf("GPU total time incl. memcpy (N=%d): %f seconds "
           "[H2D %.3f ms, kernel %.3f ms, D2H %.3f ms]\n",
           N, total_s, ms_h2d, ms_kernel, ms_d2h);
    printf("Correctness: %s (max relative error %.2e over 100 samples)\n",
           bad == 0 ? "PASS" : "FAIL", max_rel);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return bad == 0 ? 0 : 1;
}
