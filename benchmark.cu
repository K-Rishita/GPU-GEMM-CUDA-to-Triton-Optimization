#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 16

// ------------------------------------------------------------------
// Same two kernels as before, copied in here so this file is
// self-contained and just runs the benchmark.
// ------------------------------------------------------------------

__global__ void matmulNaive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulTiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
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

// ------------------------------------------------------------------
// GFLOPS calculation:
// A matmul of (MxK)*(KxN) does M*N*K multiply-adds.
// Each multiply-add = 2 floating point ops (1 multiply + 1 add).
// So total FLOPs = 2 * M * N * K.
// GFLOPS = FLOPs / (time_in_seconds * 1e9)
// ------------------------------------------------------------------
double computeGFLOPS(int M, int N, int K, float milliseconds) {
    double flops = 2.0 * M * N * K;
    double seconds = milliseconds / 1000.0;
    return (flops / seconds) / 1e9;
}

int main() {
    // Larger, realistic size for benchmarking.
    int M = 2048, N = 2048, K = 2048;
    printf("Benchmarking %dx%dx%d matmul (fp32)\n\n", M, N, K);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);

    srand(42);
    fillRandom(h_A, M * K);
    fillRandom(h_B, K * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // cudaEvent_t is the correct way to time GPU work: events are
    // recorded ON the GPU's own timeline, so they capture actual
    // kernel execution time without CPU-side measurement noise.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    const int NUM_RUNS = 10; // average over multiple runs for stability

    // ---------------- Naive ----------------
    dim3 blockDim1(16, 16);
    dim3 gridDim1((N + 15) / 16, (M + 15) / 16);

    // Warmup run (first kernel launch has extra overhead, don't count it)
    matmulNaive<<<gridDim1, blockDim1>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) {
        matmulNaive<<<gridDim1, blockDim1>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float naiveMs = ms / NUM_RUNS;
    double naiveGFLOPS = computeGFLOPS(M, N, K, naiveMs);
    printf("Naive kernel:   %8.3f ms   %8.2f GFLOPS\n", naiveMs, naiveGFLOPS);

    // ---------------- Tiled ----------------
    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmulTiled<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) {
        matmulTiled<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float tiledMs = ms / NUM_RUNS;
    double tiledGFLOPS = computeGFLOPS(M, N, K, tiledMs);
    printf("Tiled kernel:   %8.3f ms   %8.2f GFLOPS\n", tiledMs, tiledGFLOPS);

    // ---------------- cuBLAS ----------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // NOTE: cuBLAS is column-major internally. For C = A*B in
    // row-major, the standard trick is to instead compute
    // C^T = B^T * A^T, which cuBLAS sees as a normal column-major
    // multiply of B and A. That's why the argument order below
    // looks swapped -- this is a widely known cuBLAS quirk, not a bug.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float cublasMs = ms / NUM_RUNS;
    double cublasGFLOPS = computeGFLOPS(M, N, K, cublasMs);
    printf("cuBLAS:         %8.3f ms   %8.2f GFLOPS\n\n", cublasMs, cublasGFLOPS);

    // ---------------- Summary ----------------
    printf("Speedup (tiled vs naive):      %.2fx\n", naiveMs / tiledMs);
    printf("Tiled as %% of cuBLAS GFLOPS:   %.1f%%\n", 100.0 * tiledGFLOPS / cublasGFLOPS);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);

    return 0;
}
