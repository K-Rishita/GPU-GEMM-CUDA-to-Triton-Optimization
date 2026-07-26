#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// NAIVE MATMUL KERNEL
// ------------------------------------------------------------------
// C = A * B
// A is (M x K), B is (K x N), C is (M x N)
//
// KEY IDEA: one CUDA thread computes exactly one element of C.
// We launch a 2D grid of threads, one per (row, col) in C.
// Each thread reads an entire row of A and an entire column of B
// STRAIGHT FROM GLOBAL MEMORY, with no reuse across threads.
//
// This is slow because:
//   - Global memory access is ~100s of cycles of latency.
//   - Adjacent threads (same row, different col) all re-read the
//     SAME row of A independently -> massive redundant traffic.
//   - Adjacent threads (same col, different row) all re-read the
//     SAME column of B independently -> same problem.
// ------------------------------------------------------------------

__global__ void matmulNaive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    // Figure out which output element (row, col) this thread owns.
    // blockIdx = which block we're in, blockDim = threads per block,
    // threadIdx = our index within the block.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: if the matrix size isn't a perfect multiple of block size,
    // some threads at the edges don't correspond to a real element.
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row][k] -> row-major index: row * K + k
            // B[k][col] -> row-major index: k * N + col
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ------------------------------------------------------------------
// Helper: fill a matrix with random floats
// ------------------------------------------------------------------
void fillRandom(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// ------------------------------------------------------------------
// Helper: CPU reference matmul, used ONLY to verify correctness on
// a small matrix. Do not use this for timing/benchmarking.
// ------------------------------------------------------------------
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
    // Small size first, just to prove correctness.
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

    // --- Allocate GPU memory ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // --- Copy input matrices from CPU (host) to GPU (device) ---
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // --- Launch configuration ---
    // 16x16 = 256 threads per block is a common safe default.
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // --- Copy result back to CPU ---
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // --- Verify against CPU reference ---
    matmulCPU(h_A, h_B, h_C_ref, M, N, K);

    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("Naive kernel max difference vs CPU reference: %f\n", maxDiff);
    printf(maxDiff < 1e-2f ? "CORRECTNESS: PASS\n" : "CORRECTNESS: FAIL\n");

    // --- Cleanup ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    return 0;
}
