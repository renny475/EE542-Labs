// EE542 Lab 6 - Tiled CUDA matrix multiplication as a shared library for Python (ctypes)
// Build (on the GPU machine):
//   nvcc -O2 -arch=sm_75 -Xcompiler -fPIC -shared matmul_lib.cu -o libmatmul.so
// Exported C functions (no C++ name mangling, so ctypes can find them by name):
//   int         gpu_matmul(const float *A, const float *B, float *C, int N, float *timings_ms)
//   const char *gpu_error_string(int code)
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// ---- Part 4 tiled kernel (handout), unchanged ----
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

// A library must never call exit(): that would kill the Python process.
// Errors are returned as cudaError_t codes instead; 0 means success.
#define TRY(call)                                    \
    do {                                             \
        err = (call);                                \
        if (err != cudaSuccess) goto cleanup;        \
    } while (0)

extern "C" {

// C = A * B for row-major N x N float32 matrices in host memory.
// If timings_ms is not NULL, it receives {H2D copy, kernel, D2H copy} in milliseconds.
int gpu_matmul(const float *A, const float *B, float *C, int N, float *timings_ms) {
    cudaError_t err = cudaSuccess;
    float *dA = NULL, *dB = NULL, *dC = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL, e2 = NULL, e3 = NULL;
    size_t size;

    if (!A || !B || !C || N <= 0 || N > 46340) return (int)cudaErrorInvalidValue;
    size = (size_t)N * N * sizeof(float);

    TRY(cudaMalloc(&dA, size));
    TRY(cudaMalloc(&dB, size));
    TRY(cudaMalloc(&dC, size));
    TRY(cudaEventCreate(&e0)); TRY(cudaEventCreate(&e1));
    TRY(cudaEventCreate(&e2)); TRY(cudaEventCreate(&e3));

    {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

        TRY(cudaEventRecord(e0));
        TRY(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
        TRY(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));
        TRY(cudaEventRecord(e1));
        matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
        TRY(cudaGetLastError());
        TRY(cudaEventRecord(e2));
        TRY(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));
        TRY(cudaEventRecord(e3));
        TRY(cudaEventSynchronize(e3));
    }

    if (timings_ms) {
        TRY(cudaEventElapsedTime(&timings_ms[0], e0, e1));
        TRY(cudaEventElapsedTime(&timings_ms[1], e1, e2));
        TRY(cudaEventElapsedTime(&timings_ms[2], e2, e3));
    }

cleanup:
    // Free everything on every path, success or failure (cudaFree(NULL) is a no-op)
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (e2) cudaEventDestroy(e2);
    if (e3) cudaEventDestroy(e3);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (int)err;
}

const char *gpu_error_string(int code) {
    return cudaGetErrorString((cudaError_t)code);
}

}  // extern "C"
