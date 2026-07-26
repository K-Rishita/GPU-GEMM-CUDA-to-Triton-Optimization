#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// TILED MATMUL KERNEL (shared memory)
// ------------------------------------------------------------------
// Same math as naive: C = A * B, A is (MxK), B is (KxN), C is (MxN).
//
// KEY IDEA: Break the K dimension into chunks of size TILE_SIZE.
// For each chunk:
//   1. Every thread in the block cooperatively loads ONE element of A
//      and ONE element of B into shared memory arrays (tiles).
//   2. __syncthreads() makes every thread WAIT until the whole tile
//      is loaded -- otherwise some threads would start computing with
//      half-loaded data.
//   3. Each thread then does its partial-sum computation using ONLY
//      the fast shared memory tile, not global memory.
//   4. Move to the next chunk of K and repeat.
//
// Net effect: instead of each thread independently pulling K values
// from slow global memory, the block pulls each value from global
// memory ONCE and every thread that needs it reads it from shared
// memory instead. Reuse factor = TILE_SIZE.
// ------------------------------------------------------------------

#define TILE_SIZE 16

__global__ void matmulTiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    // Shared memory tiles -- visible to all threads in this block only.
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles we need to sweep across the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // --- Cooperative load: each thread loads ONE element ---
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Bounds check for matrices that aren't a clean multiple of TILE_SIZE
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // Wait for ALL threads in the block to finish loading before
        // anyone starts reading the tile -- prevents race conditions.
        __syncthreads();

        // --- Compute using the fast shared memory tile ---
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait for everyone to finish reading before we overwrite
        // the tile with the next chunk on the next loop iteration.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void fillRandom(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int M = 256, N = 256, K = 256;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    srand(42);
    fillRandom(h_A, M * K);
    fillRandom(h_B, K * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matmulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    matmulCPU(h_A, h_B, h_C_ref, M, N, K);

    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("Tiled kernel max difference vs CPU reference: %f\n", maxDiff);
    printf(maxDiff < 1e-2f ? "CORRECTNESS: PASS\n" : "CORRECTNESS: FAIL\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    return 0;
}
