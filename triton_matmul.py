"""
Triton GEMM (matrix multiplication) kernel, with autotuning,
correctness verification, and a benchmark against PyTorch's
torch.matmul (which itself calls cuBLAS under the hood on GPU).

Run this in Google Colab with a GPU runtime:
    !pip install triton  (usually already installed on Colab)
    !python 06_triton_matmul.py
"""

import torch
import triton
import triton.language as tl
import time

# ------------------------------------------------------------------
# CONCEPT: Triton kernels are written from the perspective of a
# single "program" (Triton's word for what CUDA calls a thread
# BLOCK, not an individual thread). Each program is responsible for
# computing one BLOCK_M x BLOCK_N tile of the output matrix C.
#
# Inside the kernel:
#   - tl.load / tl.store move whole tiles of data at once (Triton
#     handles the underlying memory access pattern for you).
#   - tl.dot performs a tiled matrix multiply-accumulate on the
#     tiles currently held, similar in spirit to the inner loop
#     of your CUDA tiled kernel, but expressed as one call.
#   - The accumulation loop over K chunks mirrors exactly what your
#     CUDA tiled/blocked kernels did: load a K-chunk of A and B,
#     accumulate, advance, repeat.
# ------------------------------------------------------------------

# Autotuning: Triton will benchmark all of these configs the FIRST
# time the kernel runs for a given shape, then cache the fastest one.
# This is the "autotuning-driven performance" the JD language points at
# -- rather than hand-picking one tile size like we did in CUDA, we
# let the compiler search a space of tile sizes and pick the winner.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # re-tune if the problem shape changes
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Which output tile this program instance is responsible for.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for the tile of C this program computes.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers into A and B for the FIRST K-chunk. As we loop over K,
    # we advance these pointers by BLOCK_K each iteration -- this is
    # the Triton equivalent of your CUDA kernel's "for t in numTiles".
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator held in registers/fast memory, same role as
    # `threadResults[]` in your CUDA register-blocked kernel.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Mask handles the case where K isn't a clean multiple of
        # BLOCK_K -- out-of-range reads are replaced with 0 instead
        # of reading garbage memory.
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot: tiled multiply-accumulate, conceptually the same
        # inner product your CUDA kernels computed manually with
        # nested for-loops -- Triton compiles this to efficient
        # tensor-core / FMA instructions depending on the GPU.
        acc += tl.dot(a, b)

        # Advance to the next K-chunk.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write the finished tile back to C.
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Python-side wrapper: allocates output, launches the kernel."""
    assert a.shape[1] == b.shape[0], "inner dimensions must match"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Grid: one program instance per output tile. BLOCK_M/BLOCK_N come
    # from whichever autotuned config Triton selects at runtime.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def benchmark(fn, *args, num_warmup=5, num_runs=20):
    """Times a GPU function using CUDA events, same principle as the
    cudaEvent timing you used in your CUDA benchmark -- events are
    recorded on the GPU's own timeline for accurate measurement."""
    for _ in range(num_warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_runs  # ms per run


def main():
    assert torch.cuda.is_available(), "Need a GPU runtime for this."

    torch.manual_seed(42)
    M, N, K = 2048, 2048, 2048

    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # ---- Correctness check ----
    c_triton = triton_matmul(a, b)
    c_torch = torch.matmul(a, b)  # PyTorch's own GEMM, backed by cuBLAS on GPU

    max_diff = (c_triton - c_torch).abs().max().item()
    print(f"Max difference vs torch.matmul: {max_diff:.6f}")
    print("CORRECTNESS:", "PASS" if max_diff < 1e-1 else "FAIL")
    print()

    # ---- Benchmark ----
    triton_ms = benchmark(triton_matmul, a, b)
    torch_ms = benchmark(torch.matmul, a, b)

    flops = 2.0 * M * N * K
    triton_gflops = (flops / (triton_ms / 1000)) / 1e9
    torch_gflops = (flops / (torch_ms / 1000)) / 1e9

    print(f"Triton kernel:     {triton_ms:8.3f} ms   {triton_gflops:9.2f} GFLOPS")
    print(f"torch.matmul:      {torch_ms:8.3f} ms   {torch_gflops:9.2f} GFLOPS")
    print()
    print(f"Triton as % of torch.matmul (cuBLAS) GFLOPS: {100.0 * triton_gflops / torch_gflops:.1f}%")

    # ---- Show which config autotuning picked (best-effort; this
    # attribute name has varied slightly across Triton versions, so
    # we don't want it to crash the run after the real numbers print) ----
    try:
        best_config = matmul_kernel.best_config
        print(f"\nAutotuning selected config: {best_config}")
    except AttributeError:
        print("\n(Could not read best_config attribute on this Triton version -- "
              "numbers above are still valid.)")


if __name__ == "__main__":
    main()
