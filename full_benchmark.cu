#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 16
#define BM 64
#define BN 64
#define BK 8
#define TM 8

// ---------------- Naive ----------------
__global__ void matmulNaive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---------------- Tiled ----------------
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
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ---------------- Register-blocked ----------------
__global__ void matmulBlocked(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;
    int tid = threadRow * blockDim.x + threadCol;
    float threadResults[TM] = {0.0f};
    int numTiles = K / BK;

    for (int t = 0; t < numTiles; ++t) {
        int rowA = tid / BK, colA = tid % BK;
        As[rowA][colA] = A[(blockRow * BM + rowA) * K + (t * BK + colA)];
        int rowB = tid / BN, colB = tid % BN;
        Bs[rowB][colB] = B[(t * BK + rowB) * N + (blockCol * BN + colB)];
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float bVal = Bs[k][threadCol];
            for (int r = 0; r < TM; ++r)
                threadResults[r] += As[threadRow * TM + r][k] * bVal;
        }
        __syncthreads();
    }
    for (int r = 0; r < TM; ++r) {
        int row = blockRow * BM + threadRow * TM + r;
        int col = blockCol * BN + threadCol;
        C[row * N + col] = threadResults[r];
    }
}

void fillRandom(float* mat, int size) {
    for (int i = 0; i < size; ++i) mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

double computeGFLOPS(int M, int N, int K, float milliseconds) {
    double flops = 2.0 * M * N * K;
    double seconds = milliseconds / 1000.0;
    return (flops / seconds) / 1e9;
}

int main() {
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    const int NUM_RUNS = 10;

    // ---- Naive ----
    dim3 blockDim1(16, 16);
    dim3 gridDim1((N + 15) / 16, (M + 15) / 16);
    matmulNaive<<<gridDim1, blockDim1>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) matmulNaive<<<gridDim1, blockDim1>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float naiveMs = ms / NUM_RUNS;
    double naiveGFLOPS = computeGFLOPS(M, N, K, naiveMs);
    printf("Naive kernel:      %8.3f ms   %9.2f GFLOPS\n", naiveMs, naiveGFLOPS);

    // ---- Tiled ----
    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmulTiled<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) matmulTiled<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float tiledMs = ms / NUM_RUNS;
    double tiledGFLOPS = computeGFLOPS(M, N, K, tiledMs);
    printf("Tiled kernel:      %8.3f ms   %9.2f GFLOPS\n", tiledMs, tiledGFLOPS);

    // ---- Register-blocked ----
    dim3 blockDim3(BN, BM / TM); // (64, 8)
    dim3 gridDim3(N / BN, M / BM);
    matmulBlocked<<<gridDim3, blockDim3>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i) matmulBlocked<<<gridDim3, blockDim3>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float blockedMs = ms / NUM_RUNS;
    double blockedGFLOPS = computeGFLOPS(M, N, K, blockedMs);
    printf("Register-blocked:  %8.3f ms   %9.2f GFLOPS\n", blockedMs, blockedGFLOPS);

    // ---- cuBLAS ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float cublasMs = ms / NUM_RUNS;
    double cublasGFLOPS = computeGFLOPS(M, N, K, cublasMs);
    printf("cuBLAS:            %8.3f ms   %9.2f GFLOPS\n\n", cublasMs, cublasGFLOPS);

    // ---- Summary ----
    printf("Speedup (tiled vs naive):         %.2fx\n", naiveMs / tiledMs);
    printf("Speedup (blocked vs tiled):       %.2fx\n", tiledMs / blockedMs);
    printf("Speedup (blocked vs naive):       %.2fx\n", naiveMs / blockedMs);
    printf("Blocked as %% of cuBLAS GFLOPS:    %.1f%%\n", 100.0 * blockedGFLOPS / cublasGFLOPS);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}
