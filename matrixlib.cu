#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

#define CHECK_CUDA(call)                                                        \
do {                                                                            \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));                   \
        return;                                                                 \
    }                                                                           \
} while (0)

__global__ void matrixMultiplyTiled(
    const float *A,
    const float *B,
    float *C,
    int N
)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; m++) {
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;

        tileA[ty][tx] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

        tileB[ty][tx] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gpumatrixmultiply(
    const float *h_A,
    const float *h_B,
    float *h_C,
    int N
)
{
    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return;
    }

    size_t size = (size_t)N * N * sizeof(float);

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
