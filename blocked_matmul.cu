#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// REGISTER-BLOCKED MATMUL KERNEL (1D thread coarsening)
// ------------------------------------------------------------------
// Builds on the tiled kernel's idea (shared memory tiles), but adds
// one more level: each thread now computes TM=8 output elements
// instead of just 1, holding its running sums in REGISTERS (the
// fastest memory on the chip -- faster even than shared memory).
//
// Block computes a BM x BN tile of C (64x64).
// Each thread computes TM=8 elements stacked in the M direction.
// So threads per block = (BM/TM) * BN = 8 * 64 = 512.
//
// Why this helps: in the plain tiled kernel, each value pulled from
// shared memory is used in exactly ONE multiply-add. Here, each
// value pulled from shared memory (specifically, each B value) gets
// reused across TM=8 multiply-adds via the register accumulator
// array. More compute per memory access = better use of the GPU's
// actual bottleneck (memory bandwidth), not just correctness.
//
// NOTE: for simplicity/time, this version assumes M, N, K are clean
// multiples of BM, BN, BK. That's a common simplification for a fast
// benchmark/prototype -- production kernels add boundary handling.
// ------------------------------------------------------------------

#define BM 64   // block tile height (rows of C per block)
#define BN 64   // block tile width  (cols of C per block)
#define BK 8    // depth of K we load into shared memory per step
#define TM 8    // outputs each thread computes (in the M direction)

__global__ void matmulBlocked(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Which BMxBN tile of C this block is responsible for.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Shared memory tiles for this iteration's chunk of A and B.
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread layout: blockDim = (BN, BM/TM) = (64, 8) = 512 threads.
    // threadCol picks which of the 64 output columns this thread owns.
    // threadRow picks which GROUP of TM=8 rows this thread owns.
    int threadCol = threadIdx.x;              // 0..63
    int threadRow = threadIdx.y;               // 0..7

    // Flat thread id, used only for cooperative loading below.
    int tid = threadRow * blockDim.x + threadCol; // 0..511

    // Registers: each thread's running sums for its TM output elements.
    float threadResults[TM] = {0.0f};

    int numTiles = K / BK; // assumes clean division, see note above

    for (int t = 0; t < numTiles; ++t) {
        // --- Cooperative load of As (BM x BK = 64x8 = 512 elements) ---
        // 512 threads, 512 elements -> each thread loads exactly one.
        int rowA = tid / BK;
        int colA = tid % BK;
        As[rowA][colA] = A[(blockRow * BM + rowA) * K + (t * BK + colA)];

        // --- Cooperative load of Bs (BK x BN = 8x64 = 512 elements) ---
        int rowB = tid / BN;
        int colB = tid % BN;
        Bs[rowB][colB] = B[(t * BK + rowB) * N + (blockCol * BN + colB)];

        __syncthreads();

        // --- Compute: this is where the reuse actually happens ---
        for (int k = 0; k < BK; ++k) {
            // Pull ONE value of B into a register -- shared across
            // all TM=8 multiply-adds below instead of re-reading it.
            float bVal = Bs[k][threadCol];

            for (int r = 0; r < TM; ++r) {
                float aVal = As[threadRow * TM + r][k];
                threadResults[r] += aVal * bVal;
            }
        }

        __syncthreads();
    }

    // --- Write results back to global memory ---
    for (int r = 0; r < TM; ++r) {
        int row = blockRow * BM + threadRow * TM + r;
        int col = blockCol * BN + threadCol;
        C[row * N + col] = threadResults[r];
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
    // Must be clean multiples of BM=64, BN=64, BK=8 for this kernel.
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

    dim3 blockDim(BN, BM / TM); // (64, 8) = 512 threads
    dim3 gridDim(N / BN, M / BM);

    matmulBlocked<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
    printf("Blocked kernel max difference vs CPU reference: %f\n", maxDiff);
    printf(maxDiff < 1e-2f ? "CORRECTNESS: PASS\n" : "CORRECTNESS: FAIL\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    return 0;
}
