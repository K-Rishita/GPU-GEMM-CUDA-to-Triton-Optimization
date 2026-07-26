# GPU-GEMM-CUDA-to-Triton-Optimization

A hands on project exploring GPU performance optimization, implementing matrix multiplication (GEMM, short for General Matrix Multiply) at increasing levels of sophistication in CUDA C++ and Triton, benchmarked against NVIDIA's cuBLAS.
 
## Why this project
 
Matrix multiplication is the computational core of almost every deep learning workload. Building it up from a naive implementation to something closer to production quality, and understanding why each step helps, is a good way to actually learn the GPU memory hierarchy instead of just calling an existing library.
 
## Results
 
Benchmarked on a 2048x2048x2048 fp32 matrix multiplication (NVIDIA T4 GPU, Google Colab), averaged over multiple runs using CUDA event timing.
 
| Implementation | Time (ms) | GFLOPS | % of cuBLAS |
|---|---|---|---|
| Naive CUDA | 42.231 | 406.80 | 6.3% |
| Tiled CUDA (shared memory) | 25.560 | 672.15 | 10.4% |
| Register blocked CUDA | 9.101 | 1887.65 | 29.1% |
| Triton (autotuned) | 5.224 | 3288.86 | 82.9% |
| cuBLAS / torch.matmul | 2.652 / 4.333 | 6479.21 / 3965.30 | 100% |
 
Note: cuBLAS was measured directly through cublasSgemm in the CUDA suite and indirectly through torch.matmul in the Triton suite. Both call the same underlying library, small differences reflect the measurement context.
 
Speedups: tiled is 1.65x over naive, register blocked is 2.81x over tiled and 4.64x over naive.

## Files
 
| File | Description |
|---|---|
| naive_matmul.cu | Naive kernel plus correctness check |
| tiled_matmul.cu | Shared memory tiled kernel plus correctness check |
| benchmark.cu | Naive vs tiled vs cuBLAS |
| blocked_matmul.cu | Register blocked kernel plus correctness check |
| full_benchmark.cu | Full comparison: naive, tiled, blocked, cuBLAS |
| triton_matmul.py | Autotuned Triton kernel, correctness check, benchmark vs torch.matmul |
 
## How to run
 
CUDA files need an NVIDIA GPU and nvcc (Google Colab with a GPU runtime works well).
 
```bash
# CUDA kernels
nvcc naive_matmul.cu -o naive && ./naive
nvcc tiled_matmul.cu -o tiled && ./tiled
nvcc benchmark.cu -lcublas -o benchmark && ./benchmark
nvcc blocked_matmul.cu -o blocked && ./blocked
nvcc full_benchmark.cu -lcublas -o full_bench && ./full_bench
 
# Triton kernel
pip install triton
python 06_triton_matmul.py
```
 
Each CUDA file checks correctness against a small CPU reference before trusting any performance numbers. The Triton script checks against torch.matmul directly.
