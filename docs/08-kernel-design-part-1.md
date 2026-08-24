# Kernel Design, Warp Specialization & Pipelining
## Lesson 3 Companion Guide — High-Performance CUDA Kernels on the H100

---

## ★ BEFORE YOU START — Prerequisites & Setup

### What You Need to Know First

This guide builds directly on Lesson 2 (Clusters, Inline PTX, Data Types & Pointers). You should be comfortable with:

```
Prerequisites Checklist
══════════════════════════════════════════════════════════════════════

  From Lesson 2:
    ✓ Thread Block Clusters and cooperative_groups API
    ✓ Inline PTX assembly (asm() syntax, constraints, clobbers)
    ✓ State spaces (.shared, .global, .reg)
    ✓ Data types (FP16, BF16, FP8, packed types)
    ✓ cvta, mapa — shared memory address translation

  New concepts introduced HERE (no prior knowledge needed):
    ★ Arithmetic Intensity and the Roofline Model
    ★ Compute-bound vs Memory-bound kernel classification
    ★ Warp Specialization (producer/consumer warp groups)
    ★ setmaxnreg — dynamic register allocation
    ★ Circular buffering with mbarrier synchronization
    ★ TMA pipeline mechanics (full/empty barriers)
    ★ Ping-pong and cooperative pipeline architectures
    ★ Persistent scheduling and tile rasterization
    ★ Stream-K scheduling
    ★ Epilogue fusion

  Hardware targets (same as Lesson 2):
    H100 (sm_90a)  — all sections apply
    A100 (sm_80)   — Sections 1-2, 13-16 (no wgmma, limited barriers)
    B100 (sm_100a) — all sections, same code, different compile flag
```

### Compile Commands for This Guide

Every example uses one of these three compile patterns:

```
Compile Patterns
══════════════════════════════════════════════════════════════════════

  # Standard (most examples — no Hopper-specific features)
  nvcc -std=c++17 -arch=sm_90a -O2 example.cu -o example

  # With cooperative groups (cluster/warp group examples)
  nvcc -std=c++17 -arch=sm_90a -O2 \
       -I/usr/local/cuda/include example.cu -o example

  # For B100/Blackwell (drop-in replacement)
  nvcc -std=c++17 -arch=sm_100a -O2 example.cu -o example

  # Show register/SMEM usage (always useful for optimization)
  nvcc -std=c++17 -arch=sm_90a -O2 --ptxas-options=-v example.cu -o example
  # Look for:
  #   "Used N registers" → occupancy impact
  #   "spill stores/loads: N bytes" → register pressure too high!
```

---

## Table of Contents

1. [The Roofline Model — Classifying Your Kernel](#1--the-roofline-model)
2. [Compute-Bound Kernels — Characteristics & Diagnosis](#2--compute-bound-kernels)
3. [Memory-Bound Kernels — Characteristics & Diagnosis](#3--memory-bound-kernels)
4. [Warp Specialization — Splitting Producer and Consumer](#4--warp-specialization)
5. [setmaxnreg — Dynamic Register Allocation](#5--setmaxnreg)
6. [Warp Group Count Selection](#6--warp-group-count-selection)
7. [Pipelining — The Fundamental Overlap Principle](#7--pipelining)
8. [Circular Buffering and Multi-Stage Pipelines](#8--circular-buffering)
9. [mbarrier — The Two-Barrier Handshake](#9--mbarrier)
10. [ABA Ambiguity and Phase Bits](#10--aba-ambiguity-and-phase-bits)
11. [Producer Flow — The Full TMA Pipeline Loop](#11--producer-flow)
12. [Consumer Flow — WGMMA Pipelining](#12--consumer-flow)
13. [The Cooperative Pipeline Architecture](#13--cooperative-pipeline)
14. [The Ping-Pong Pipeline Architecture](#14--ping-pong-pipeline)
15. [Epilogue Fundamentals — From Registers to Global Memory](#15--epilogue-fundamentals)
16. [Epilogue Fusion — The Six-Stage Lifecycle](#16--epilogue-fusion)
17. [Persistent Scheduling — Static, Grouped, Stream-K](#17--persistent-scheduling)
18. [Rasterization and L2 Cache Locality](#18--rasterization)
19. [Putting It All Together — Full Worked Examples](#19--putting-it-all-together)
20. [Quick Reference](#20--quick-reference)

---

## 1 — The Roofline Model

### 1.1 Why Every Kernel Is Either Compute-Bound or Memory-Bound

Every GPU kernel has exactly one bottleneck: either the hardware is waiting for data
to arrive (memory-bound), or the hardware is waiting for arithmetic units to finish
(compute-bound). The roofline model gives you the mathematical tool to determine
which one you are — and how far from peak performance you currently sit.

The key quantity is **Arithmetic Intensity (AI)**: how many floating-point operations
you perform per byte of data you move between the memory hierarchy and the compute
units.

```
The Roofline Model — Visual and Equations
══════════════════════════════════════════════════════════════════════

  Arithmetic Intensity (AI):
    AI = Total FLOPs executed / Total bytes moved from/to HBM
    Units: FLOP/Byte

  The Ridge Point (crossover):
    Ridge Point = Peak FLOPS / Peak Bandwidth
    H100 SXM5 example:
      FP16 Tensor Core FLOPS: 989 TFLOPS
      HBM3 Bandwidth:         3.35 TB/s
      Ridge (FP16 Tensor):    989 / 3.35 ≈ 295 FLOP/Byte
      Ridge (FP32 CUDA core): ~67 TFLOPS / 3.35 ≈ 20 FLOP/Byte

  The Roofline Chart:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Performance │                                                   │
  │ (TFLOPS)    │                         ╔══════════════════════   │
  │   989 ──────┼────────────────────────═╝ ← Compute Roof (FP16) │
  │             │              ╔══════════════════                  │
  │             │             /          compute-bound region       │
  │             │            / ← slope = bandwidth (3.35 TB/s)     │
  │             │           /                                       │
  │             │   memory-bound region                             │
  │       0 ────┼──────────────────────────────────────────────────│
  │             │         ↑               ↑                         │
  │             │      AI=20           AI=295                       │
  │             │  (FP32 ridge)    (FP16 ridge)                     │
  │             └─────────────────────────────────────────────────  │
  │                         Arithmetic Intensity (FLOP/Byte)        │
  └─────────────────────────────────────────────────────────────────┘

  Reading the chart:
    • If your kernel's AI < ridge point → memory-bound
      Performance ∝ AI × bandwidth (more intensity = more throughput)
    • If your kernel's AI > ridge point → compute-bound
      Performance ∝ Peak FLOPS (more intensity won't help)

  KEY INSIGHT: The roofline tells you WHERE to focus your optimization.
    Memory-bound? Reduce bytes moved (tiling, caching, fusion).
    Compute-bound? Maximize FLOPs per SM cycle (warp specialization, pipelines).
```

### 1.2 Computing AI for Real Operations

Let's calculate the AI for common deep learning operations. This tells you immediately which class each operation falls into.

```
Arithmetic Intensity — Common Operations
══════════════════════════════════════════════════════════════════════

  Operation         FLOPs                Bytes (FP16)  AI          Class
  ─────────────────────────────────────────────────────────────────────
  elementwise relu  N (one comparison)   2N (read+write)  0.5       Memory
  layer norm        ~5N                  ~4N              ~1.25     Memory
  softmax           ~5N                  ~4N              ~1.25     Memory
  bias add          N                    3N               0.33      Memory
  GEMM (M=N=K=4096) 2×4096³ ≈ 137G FLOP 3×4096²×2=100MB  ~1370    Compute
  GEMM (M=N=K=256)  2×256³  ≈ 33M FLOP  3×256²×2=400KB   ~83      Compute
  GEMM (M=1, N=K=4096) 2×4096² ≈ 33M   ~34MB             ~1       Memory!
  Attention (seq=2048, d=64) complex     complex           ~10-50   Compute

  INTERPRETATION:
    GEMM with large, square matrices → deep into compute-bound territory
    GEMM with tiny batch or tiny M → memory-bound (vector-matrix mult)
    All pointwise ops (relu, gelu, etc.) → heavily memory-bound
    Attention → context-dependent, can be either

  ─── Why This Matters ────────────────────────────────────────────
  If you spend 3 days optimizing the arithmetic throughput of a
  softmax kernel (AI ≈ 1.25), you're optimizing the WRONG thing.
  The bottleneck is bandwidth. Fuse it into the preceding GEMM
  (which is compute-bound) instead. This is why epilogue fusion
  (Section 15-16) is so valuable — it eliminates entire kernels'
  worth of bandwidth by merging them into the GEMM epilogue.

  Computing AI for your own kernel:
  ─────────────────────────────────────────────────────────────────
  Method 1: Nsight Compute → "roofline" section → reads AI directly
  Method 2: Calculate manually:
    FLOPs = count arithmetic ops in kernel × number of elements
    Bytes = (reads from global) + (writes to global), each in bytes
    AI = FLOPs / Bytes
  Method 3: Use the formula for GEMM:
    AI_GEMM = 2MNK / (2×(MK + NK + MN)) bytes
    For square matrices (M=N=K=n): AI = 2n³ / (6n²×2) = n/6 FLOP/Byte
    At n=4096: AI = 4096/6 ≈ 683 FLOP/Byte (assuming FP32 storage)
```

---

### ★ Section 1 — Runnable Code: Roofline Profiler

**Manual Worked Example:**

Let's manually compute AI for the test kernels below.

```
Kernel A: elementwise_scale (multiply each element by a scalar)
  Input: N=1M float32 values = 4 MB read
  Output: N=1M float32 values = 4 MB write
  Total bytes: 8 MB
  FLOPs: N multiplications = 1M FLOPs
  AI = 1M / 8M = 0.125 FLOP/Byte → heavily memory-bound

Kernel B: matrix_multiply_naive (M=N=K=512)
  Input: A matrix (512×512 × 4 bytes) = 1MB, B matrix = 1MB
  Output: C matrix = 1MB
  Total bytes: 3 MB
  FLOPs: 2 × 512³ = 268M FLOPs
  AI = 268M / 3M = ~89 FLOP/Byte → compute-bound (above FP32 ridge of ~20)

Expected profiler output:
  elementwise_scale: AI ≈ 0.125 → MEMORY-BOUND
  matrix_multiply:   AI ≈ 89    → COMPUTE-BOUND
```

```
═══════════════════════════════════════════════════════
  FILE:    ex01_roofline.cu
  COMPILE: nvcc -std=c++17 -arch=sm_90a -O2 ex01_roofline.cu -o ex01_roofline
           (Also works on sm_80 — no Hopper-specific features)
  RUN:     ./ex01_roofline
  EXPECT:  AI values with memory-bound/compute-bound classification,
           measured vs theoretical bandwidth and FLOPS.
═══════════════════════════════════════════════════════
```

```cuda
// ═══════════════════════════════════════════════════════════════════
// ex01_roofline.cu
//
// PURPOSE: Measure actual bandwidth and FLOP rate of two kernels
//   to demonstrate the Roofline model in practice.
//
// KERNELS:
//   Kernel A: elementwise_scale  — memory-bound (AI ≈ 0.125)
//   Kernel B: tile_matmul        — compute-bound (AI >> 20)
//
// WHAT TO LOOK FOR:
//   - Kernel A achieves close to peak memory bandwidth
//   - Kernel A is far from peak FLOPS (memory limited)
//   - Kernel B achieves higher FLOPS, limited by compute
//   - Both kernels are described by the roofline
//
// WORKS ON: sm_80, sm_86, sm_89, sm_90a, sm_100a
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>

// For timing
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// ─── Timing helper ─────────────────────────────────────────────────
// Returns elapsed time in milliseconds between two CUDA events.
// Usage:
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start); cudaEventCreate(&stop);
//   cudaEventRecord(start);
//   ... kernel launch ...
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   float ms = elapsed_ms(start, stop);
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL A: Elementwise Scale — Memory-Bound
// ═══════════════════════════════════════════════════════════════════
//
// Each thread reads one float, multiplies by scale, writes one float.
// AI = 1 multiply / (4+4 bytes) = 0.125 FLOP/Byte
//
// This kernel is entirely limited by HBM bandwidth.
// No matter how fast the FP32 ALUs are, the bottleneck is moving
// the data from HBM to registers and back.
__global__ void elementwise_scale(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // One load, one multiply, one store.
        // The __restrict__ qualifier tells the compiler these pointers
        // don't alias, enabling better scheduling.
        output[idx] = input[idx] * scale;
    }
}

// ═══════════════════════════════════════════════════════════════════
// KERNEL B: Tiled Matrix Multiply — Compute-Bound
// ═══════════════════════════════════════════════════════════════════
//
// Computes C = A × B using shared memory tiling.
// For TILE=32, M=N=K=512:
//   FLOPs = 2 × 512³ = 268M
//   Bytes = (512²+512²+512²) × 4 = 3MB
//   AI ≈ 89 FLOP/Byte → well above FP32 ridge (~20 FLOP/Byte)
//
// The tiling pattern (load tile to SMEM, compute from SMEM) is
// the classic way to improve AI: each element of A and B is loaded
// TILE times instead of once, multiplying the AI by TILE.
//
// ─── Why AI improves with tiling ────────────────────────────────
// Without tiling: AI = 2K / (K + K) * ...(complex)... ≈ N/2 for square
// With TILE tiling: each element of A is loaded once per K/TILE steps
//   and reused TILE times from SMEM → TILE× better bandwidth efficiency
#define TILE 32

__global__ void tile_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // ── Shared memory tiles ─────────────────────────────────────
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;  // accumulator stays in register throughout

    // ── K-loop: load tiles and accumulate ───────────────────────
    // Each iteration loads a TILE×TILE tile of A and B into SMEM,
    // then does TILE² multiply-add operations from SMEM (cheap).
    // After K/TILE iterations, acc holds the full dot product.
    for (int t = 0; t < K / TILE; t++) {
        // Collaborative load: each thread loads one element of A and B
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // wait for all threads to finish loading

        // ── Compute from SMEM (cheap, ~10 cycle latency) ────────
        // TILE multiply-adds per element, using the shared tile.
        // These run from SMEM, not HBM — ~10 cycle vs ~500 cycle.
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // wait before loading next tile
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Roofline Model Demo — Memory vs Compute Bound        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ── Get GPU specs ─────────────────────────────────────────────
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("     Peak memory bandwidth (theoretical): %.1f GB/s\n",
           (double)prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6);
    printf("     SM count: %d\n\n", prop.multiProcessorCount);

    // ── Kernel A: Elementwise Scale ───────────────────────────────
    printf("════════════════════════════════════════════════════════\n");
    printf("KERNEL A: elementwise_scale (expected: MEMORY-BOUND)\n");
    printf("════════════════════════════════════════════════════════\n");
    {
        const int N = 64 * 1024 * 1024;  // 64M elements = 256MB
        const float SCALE = 2.5f;

        // Theoretical AI = 1 FLOPs / 8 bytes = 0.125 FLOP/Byte
        double ai_theoretical = 1.0 / 8.0;
        double flops_total = (double)N;
        double bytes_total = 2.0 * N * sizeof(float);  // read + write

        printf("  Elements: %dM    Bytes moved: %.0f MB    AI: %.3f FLOP/Byte\n",
               N / (1024*1024), bytes_total / 1e6, ai_theoretical);
        printf("  (Ridge point for FP32 CUDA cores ≈ 20 FLOP/Byte)\n");
        printf("  → AI=%.3f << 20 → MEMORY-BOUND ✓\n\n", ai_theoretical);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        // Initialize (don't care about values for timing purposes)
        CHECK_CUDA(cudaMemset(d_in, 0, N * sizeof(float)));

        // Warm up the GPU (avoids one-time initialization in timing)
        elementwise_scale<<<N/256, 256>>>(d_in, d_out, SCALE, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // ── Timed run ──────────────────────────────────────────
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        const int REPS = 10;
        cudaEventRecord(start);
        for (int i = 0; i < REPS; i++)
            elementwise_scale<<<N/256, 256>>>(d_in, d_out, SCALE, N);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaDeviceSynchronize());
        float ms = elapsed_ms(start, stop) / REPS;

        double bw_achieved  = bytes_total / (ms / 1000.0) / 1e9;  // GB/s
        double gflops_achieved = flops_total / (ms / 1000.0) / 1e9; // GFLOPS

        printf("  Time: %.3f ms\n", ms);
        printf("  Bandwidth achieved: %.1f GB/s\n", bw_achieved);
        printf("  GFLOPS achieved:    %.1f\n", gflops_achieved);
        printf("  → Kernel is BANDWIDTH limited (not compute limited)\n");

        cudaEventDestroy(start); cudaEventDestroy(stop);
        CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    }

    // ── Kernel B: Tiled Matrix Multiply ──────────────────────────
    printf("\n════════════════════════════════════════════════════════\n");
    printf("KERNEL B: tile_matmul (expected: COMPUTE-BOUND)\n");
    printf("════════════════════════════════════════════════════════\n");
    {
        const int M = 512, N = 512, K = 512;

        double flops_total = 2.0 * M * N * K;  // multiply + add per element
        double bytes_total = ((double)M*K + (double)K*N + (double)M*N) * sizeof(float);
        double ai = flops_total / bytes_total;

        printf("  M=N=K=%d    FLOPs: %.1f M    Bytes: %.1f MB    AI: %.1f FLOP/Byte\n",
               M, flops_total/1e6, bytes_total/1e6, ai);
        printf("  → AI=%.1f >> 20 → COMPUTE-BOUND ✓\n\n", ai);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_A, 1, M * K * sizeof(float)));  // fill with 0x01
        CHECK_CUDA(cudaMemset(d_B, 1, K * N * sizeof(float)));

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        // Warm up
        tile_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        const int REPS = 20;
        cudaEventRecord(start);
        for (int i = 0; i < REPS; i++)
            tile_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        CHECK_CUDA(cudaDeviceSynchronize());
        float ms = elapsed_ms(start, stop) / REPS;

        double gflops_achieved = flops_total / (ms / 1000.0) / 1e9;
        double bw_achieved     = bytes_total / (ms / 1000.0) / 1e9;

        printf("  Time: %.3f ms\n", ms);
        printf("  GFLOPS achieved: %.1f\n", gflops_achieved);
        printf("  Effective BW:    %.1f GB/s (not the bottleneck)\n", bw_achieved);
        printf("  → Kernel is COMPUTE limited (bandwidth not saturated)\n");

        cudaEventDestroy(start); cudaEventDestroy(stop);
        CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAY:                                        ║\n");
    printf("║  • elementwise_scale: optimize bandwidth, not FLOPs   ║\n");
    printf("║  • tile_matmul: optimize FLOPs, pipeline depth        ║\n");
    printf("║  • Wrong optimization → wrong results → wasted time   ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    return 0;
}
```

---

## 2 — Compute-Bound Kernels

### 2.1 The Three Signatures of a Compute-Bound Kernel

When your kernel's AI exceeds the ridge point, arithmetic throughput becomes the bottleneck. But recognizing a compute-bound kernel in practice — before you profile — requires understanding what the hardware tells you.

```
Compute-Bound Signatures — What to Look For
══════════════════════════════════════════════════════════════════════

  SIGNATURE 1: High Arithmetic Intensity
  ─────────────────────────────────────────────────────────────────
  AI > ridge point (>295 FLOP/Byte for FP16 Tensor on H100)
  The kernel does many arithmetic operations per byte it loads.
  Example: GEMM with large tiles, attention with many heads.

  SIGNATURE 2: Producer/Consumer Warp Imbalance
  ─────────────────────────────────────────────────────────────────
  The most DIAGNOSTIC signature of a compute-bound kernel.
  When you look at the kernel timeline in Nsight Compute, you see:

  Time →
  Producer warps: [LOAD][LOAD][LOAD][WAIT.....][LOAD][LOAD][WAIT...]
  Consumer warps: [MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA]

  The producer warps are idle (waiting at barriers) while consumer
  warps are 100% busy doing WGMMA.
  If producers are blocking, that means consumers can't keep up
  — the math is the bottleneck.

  SIGNATURE 3: High SM Activity, Low Memory Transactions
  ─────────────────────────────────────────────────────────────────
  Nsight Compute → Memory Workload Analysis:
    "Memory Throughput" = low % of peak
    "Compute Throughput" = high % of peak
  The kernel is not bottlenecked on HBM reads/writes.

  ─── How to Classify Using Nsight ────────────────────────────────
  1. Profile with: ncu --metrics all ./my_kernel
  2. Look for "achieved_occupancy" and "sm__throughput.avg.pct_of_peak"
  3. Look for "l1tex__t_bytes.sum" and "dram__bytes.sum"
  4. If dram utilization is low but SM utilization is high → compute-bound

  ─── Common Compute-Bound Kernels ────────────────────────────────
  ├── GEMM with large M,N,K (the king of compute-bound)
  ├── Batched GEMM with sufficient batch size
  ├── Flash Attention 2/3 (for large sequence lengths)
  ├── Convolutions with large channel counts
  └── Any kernel that streams data into tensor cores

  ─── The Key Question for Optimization ───────────────────────────
  "Are my tensor cores busy 100% of the time?"
  If NO → your job is to eliminate everything that prevents this.
    Sources of idle time: barrier stalls, memory stalls, branch divergence,
    producer/consumer imbalance, insufficient pipelining.
  If YES → you're at (or near) the hardware limit. Time to choose
    between better algorithms or accept the limit.
```

### 2.2 Optimization Techniques Overview

For compute-bound kernels, there are seven major optimization strategies. Each is covered in depth in the sections that follow.

```
Compute-Bound Optimization Hierarchy
══════════════════════════════════════════════════════════════════════

  Priority  Technique                  Sections  Impact
  ─────────────────────────────────────────────────────────────────
  1 (HIGH)  Warp Specialization        4-6       Decouples loads & math
  2 (HIGH)  Pipelining (multi-stage)   7-8       Hides memory latency
  3 (HIGH)  mbarrier synchronization   9-10      Efficient producer-consumer
  4 (MED)   Pipeline Architecture      13-14     Ping-pong vs cooperative
  5 (MED)   Cluster TMA multicast      Lesson 2  Reduces HBM bandwidth
  6 (LOW)   Epilogue Fusion            15-16     Eliminates separate kernels
  7 (LOW)   Persistent Scheduling      17-18     Eliminates wave quantization

  ─── What NOT to Optimize for Compute-Bound Kernels ─────────────
  ✗ Bank conflicts (usually not the bottleneck)
  ✗ Global memory coalescing (already high AI means few global reads)
  ✗ Occupancy optimization (more threads may hurt register availability)
  ✗ Removing arithmetic operations (you're compute-bound already!)

  The goal is simple: keep the tensor cores busy 100% of the time.
  Everything else is in service of that goal.
```

---

## 3 — Memory-Bound Kernels

### 3.1 The Three Memory-Bound Categories

Memory-bound is not a single condition. The bottleneck can be at three different points in the memory hierarchy, each with different fixes.

```
Three Categories of Memory-Bound Kernels
══════════════════════════════════════════════════════════════════════

  ─── Category 1: Bandwidth-Bound ─────────────────────────────────
  Symptom: DRAM throughput near peak (>80% of HBM bandwidth)
  Cause: The kernel moves many bytes relative to its FLOPs.
  Examples: elementwise ops, data type conversions, reductions,
            layer norm, softmax.
  Fix:
    ├── Epilogue fusion — merge the op into an adjacent GEMM
    ├── Operator fusion — combine multiple ops into one kernel
    ├── Vector loads — use float4 instead of float (wider transactions)
    └── Reduce bytes moved — use smaller data types (FP16 vs FP32)

  ─── Category 2: Latency-Bound ───────────────────────────────────
  Symptom: Low DRAM throughput AND low FLOP rate. Hardware is "stuck".
  Cause: Long dependency chains (each operation waits for the previous).
         OR: synchronization overhead (barriers, atomics, fences).
  Examples: Reduction kernels with poor tree structure, kernels with
            many dependent memory accesses (pointer-following).
  Fix:
    ├── Increase thread-level parallelism (more independent operations)
    ├── Software prefetching (issue loads early before data is needed)
    ├── Warp-level primitives instead of shared memory for reductions
    └── Avoid __syncthreads() where possible (use warp intrinsics)

  ─── Category 3: Locality-Bound ──────────────────────────────────
  Symptom: High DRAM transactions but low bandwidth utilization.
           Many cache misses despite seemingly reasonable access patterns.
  Cause: Data accessed in random order, cache thrashing, TLB pressure,
         working set larger than L2 cache.
  Examples: Embedding lookups, sparse ops, gather/scatter.
  Fix:
    ├── Reorder data in memory to match access pattern
    ├── Blocking/tiling to fit working set in cache
    ├── Swizzling (covered in Section 18)
    └── Reduce working set size

  ─── Diagnosis Flowchart ──────────────────────────────────────────

  Start: Is HBM bandwidth > 80% of peak?
    YES → Bandwidth-bound → fuse or reduce bytes
    NO  → Is FLOP rate > 20% of peak?
      YES → partially compute-bound → increase occupancy or AI
      NO  → Is SM utilization > 50%?
        YES → Latency-bound → break dependency chains
        NO  → Locality-bound → improve cache utilization
```

### 3.2 The Epilogue as the Universal Memory-Bound Killer

In a real training pipeline, the operations after GEMM (bias, activation, scaling) are all memory-bound. Instead of running them as separate kernels (3+ HBM round-trips), they are fused into the GEMM epilogue (0 extra HBM round-trips — the data stays in registers).

```
Without Fusion vs With Fusion — Bandwidth Analysis
══════════════════════════════════════════════════════════════════════

  Operation: GEMM → bias → ReLU → layer_norm → write output
  Matrix size: M=N=4096 (output = 4096×4096 × 4 bytes = 64 MB)

  WITHOUT FUSION (4 separate kernels):
  ─────────────────────────────────────────────────────────────────
  Kernel 1 (GEMM):       compute A×B, write 64MB to HBM
  Kernel 2 (bias add):   read 64MB from HBM, add bias, write 64MB
  Kernel 3 (ReLU):       read 64MB from HBM, apply ReLU, write 64MB
  Kernel 4 (layer norm): read 64MB from HBM, normalize, write 64MB

  Total HBM traffic: 64MB + 128MB + 128MB + 128MB = 448MB
  At 3.35 TB/s HBM bandwidth: 448MB / 3350 GB/s = 0.13ms (JUST bandwidth)
  Plus 4 kernel launches, 4 sets of cache invalidations, etc.

  WITH EPILOGUE FUSION (1 kernel):
  ─────────────────────────────────────────────────────────────────
  Kernel 1 (GEMM+epilogue): compute A×B in registers
    → apply bias IN REGISTERS (0 HBM reads)
    → apply ReLU IN REGISTERS (0 HBM reads)
    → apply layer norm PARTIALLY in registers + SMEM
    → write 64MB final output to HBM

  Total HBM traffic for the fused parts: 64MB (only the final write)
  That's 7× less HBM traffic for the epilogue portion!

  RULE: If an operation runs immediately after a GEMM and its
        output is consumed by a later GEMM, it should be in
        the epilogue, not a separate kernel.
```

---

## 4 — Warp Specialization

### 4.1 The Core Concept — Intentional Role Division

On any GPU, an SM hosts many active warps, but only a few can issue instructions each cycle. The default scheduling model lets all warps do the same type of work (e.g., all warps do loads AND math). This creates competition: warps doing math occupy slots that warps needing to do loads could use.

Warp Specialization breaks this competition by giving different warps **permanently different jobs** for the entire kernel execution. Some warps become dedicated "producers" that only move data. Other warps become dedicated "consumers" that only perform arithmetic.

```
Warp Specialization — Before and After
══════════════════════════════════════════════════════════════════════

  ─── WITHOUT Warp Specialization ─────────────────────────────────

  All warps do both load and compute:

  Warp 0:  [LOAD][LOAD]....wait....[MMA][MMA][LOAD][LOAD]....wait....
  Warp 1:  ...[LOAD][LOAD]....wait....[MMA][MMA][LOAD][LOAD]....wait.
  Warp 2:  ......[LOAD][LOAD]....wait....[MMA][MMA][LOAD][LOAD].....
  Warp 3:  .........[LOAD][LOAD]....wait....[MMA][MMA][LOAD]........

  Problems:
  ├── "wait" cycles: memory latency (~500 cycles) blocks math issue
  ├── Register file split: same warp needs regs for BOTH load state
  │   AND math accumulator → half the registers for each → spilling
  └── Compiler can't schedule optimally: mixed dependencies

  ─── WITH Warp Specialization ────────────────────────────────────

  Producer warps (e.g., WG0): ONLY issue loads and manage barriers
  Consumer warps (e.g., WG1, WG2): ONLY issue WGMMA instructions

  WG0 (Producer):  [TMA][TMA][TMA][TMA][TMA][TMA][TMA][TMA][TMA]
  WG1 (Consumer):  [MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA]
  WG2 (Consumer):  [MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA][MMA]

  Results:
  ├── Producer runs ~32 registers (only manages pointers/barriers)
  ├── Consumer runs ~200 registers (holds full output tile)
  ├── Both run SIMULTANEOUSLY on the same SM
  └── Memory latency HIDDEN by overlapping with compute

  ─── The Three Reasons It Works ──────────────────────────────────

  Reason 1 — Resource separation:
    A WGMMA consumer needs ~200 registers for its accumulator tile.
    A TMA producer needs ~32 registers for barrier management.
    Together on one SM: 200+32 = 232 registers total.
    Without separation: each mixed warp needs 200+32 = 232 registers,
    and with 4 warps that's 4×232 = 928 registers, crushing occupancy.
    With separation: producer + consumer share register budget efficiently.

  Reason 2 — Latency hiding by overlap:
    While WG0 waits for the TMA to fill SMEM (~400 cycles),
    WG1 is executing WGMMA on the PREVIOUS tile's data.
    No SM cycles are wasted — either loads or math is happening.

  Reason 3 — Avoiding SIMT branch divergence:
    If a single warp does if (i_am_producer) { ... } else { ... },
    the warp must execute BOTH branches serially (SIMT divergence).
    With warp specialization: producer warps ONLY execute producer code,
    consumer warps ONLY execute consumer code. Zero divergence.
```

### 4.2 Implementing Warp Specialization in Code

The key to implementing warp specialization is determining, at runtime, which warp group the current thread belongs to — and then branching into the appropriate persistent loop.

```
Warp Specialization Code Pattern
══════════════════════════════════════════════════════════════════════

  How to identify warp group membership:

  // Warp ID within the block (0 to num_warps-1)
  int warp_id = threadIdx.x / 32;

  // Warp GROUP ID (groups of 4 warps, 128 threads each)
  int warpgroup_id = warp_id / 4;

  // Total warpgroups per block
  // If block has 384 threads: 384/128 = 3 warpgroups (WG0, WG1, WG2)

  // Branch based on role
  if (warpgroup_id == 0) {
      // I am a PRODUCER warpgroup
      producer_loop(...);
  } else {
      // I am a CONSUMER warpgroup
      consumer_loop(warpgroup_id - 1, ...);  // consumer index 0 or 1
  }

  ─── The Key Dispatch Pattern ─────────────────────────────────────

  // Typical H100 GEMM warp layout:
  // 384 threads per block = 3 warpgroups:
  //   WG0 (threads 0-127):   Producer  — TMA loads
  //   WG1 (threads 128-255): Consumer0 — WGMMA + epilogue
  //   WG2 (threads 256-383): Consumer1 — WGMMA + epilogue (ping-pong)

  constexpr int THREADS_PER_WARPGROUP = 128;  // 4 warps × 32 threads

  int warpgroup_id = threadIdx.x / THREADS_PER_WARPGROUP;

  if (warpgroup_id == 0) {
      // Producer code path
      // This branch never issues WGMMA instructions
      // It only: waits on empty barriers, issues TMA cp.async.bulk,
      //          signals full barriers, manages pipeline state
  } else {
      // Consumer code path
      // This branch never issues TMA instructions
      // It only: waits on full barriers, issues WGMMA,
      //          signals empty barriers, writes epilogue output
  }

  ─── Why Branches Don't Cause Divergence ─────────────────────────
  Divergence occurs when threads in the SAME WARP take different paths.
  In warp specialization, the branch boundary aligns to WARP GROUP
  boundaries (multiples of 128 threads = 4 warps).
  WG0's warps ALL take the producer branch.
  WG1's warps ALL take the consumer branch.
  No single warp straddles the boundary → zero divergence.
```

---

### ★ Section 4 — Runnable Code: Warp Specialization Demo

**Manual Worked Example:**

We launch a kernel with 256 threads = 2 warp groups (WG0 and WG1).
- WG0 (threads 0-127): "producer" — writes a buffer with values computed from thread index
- WG1 (threads 128-255): "consumer" — reads that buffer and doubles the values

Without proper synchronization, WG1 might read before WG0 finishes. We use `__syncthreads()` here for simplicity (a real pipeline would use `mbarrier` — covered in Section 9).

```
Expected data flow:
  WG0 thread 0:  smem[0]   = 0   × 3 + 7 = 7
  WG0 thread 1:  smem[1]   = 1   × 3 + 7 = 10
  WG0 thread 63: smem[63]  = 63  × 3 + 7 = 196
  WG0 thread 127:smem[127] = 127 × 3 + 7 = 388

After sync, WG1 reads and doubles:
  WG1 thread 0 (local_idx=0): output[0] = smem[0]  × 2 = 14
  WG1 thread 1 (local_idx=1): output[1] = smem[1]  × 2 = 20
  WG1 thread 127 (local_idx=127): output[127] = 388 × 2 = 776
```

```
═══════════════════════════════════════════════════════
  FILE:    ex02_warp_spec.cu
  COMPILE: nvcc -std=c++17 -arch=sm_90a -O2 ex02_warp_spec.cu -o ex02_warp_spec
           (Also works on sm_80, sm_86, sm_89)
  RUN:     ./ex02_warp_spec
  EXPECT:  All 128 consumer outputs verified as 2×(3×idx+7),
           role assignment report showing clean WG separation.
═══════════════════════════════════════════════════════
```

```cuda
// ═══════════════════════════════════════════════════════════════════
// ex02_warp_spec.cu
//
// PURPOSE: Demonstrate warp specialization — intentional role division.
//
// LAYOUT:
//   256 threads per block = 2 warpgroups
//   WG0 (threads   0-127): PRODUCER — fills shared memory buffer
//   WG1 (threads 128-255): CONSUMER — reads buffer, doubles values
//
// WHAT TO LOOK FOR IN OUTPUT:
//   - Role assignment: each warpgroup correctly identifies itself
//   - All 128 consumer outputs match expected: 2 × (3*idx + 7)
//   - No divergence: all producer threads do exactly the same work,
//     all consumer threads do exactly the same work
//
// NOTE: In production kernels, the __syncthreads() between producer
// and consumer is replaced by mbarrier (Section 9). __syncthreads()
// is used here for clarity.
//
// WORKS ON: sm_80+ (no Hopper-specific features)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// Size of each warpgroup (4 warps × 32 threads = 128 threads)
constexpr int WARPGROUP_SIZE = 128;

// ── Producer function ─────────────────────────────────────────────
// Only called by WG0 threads.
// Fills shared memory with a pattern: value = local_idx * 3 + 7
// This simulates loading data from global memory / TMA.
__device__ void producer_work(float* smem, int local_idx, int wg_size) {
    // Each producer thread writes its own slot
    float val = local_idx * 3.0f + 7.0f;
    smem[local_idx] = val;

    // In a real pipeline, this is where you'd issue TMA and
    // signal the full barrier. Here, the global __syncthreads()
    // in the main kernel serves as our barrier.
}

// ── Consumer function ─────────────────────────────────────────────
// Only called by WG1 threads.
// Reads from shared memory (written by producer) and doubles it.
// local_idx = thread index within WG1 (0 to 127).
__device__ float consumer_work(const float* smem, int local_idx) {
    // Read from SMEM (populated by producer) and compute
    float val = smem[local_idx];
    return val * 2.0f;  // "process" the data
}

// ── Main kernel ───────────────────────────────────────────────────
// 256 threads: WG0 produces, WG1 consumes.
// Shared memory buffer is the communication channel.
__global__ void warp_specialized_kernel(
    float* output,         // output array [128 floats]
    int*   role_record)    // records which WG each thread belongs to
{
    // Shared buffer: WG0 writes, WG1 reads.
    // Size = 1 warpgroup worth of data (128 floats)
    __shared__ float smem_buffer[WARPGROUP_SIZE];

    // ── Determine role ───────────────────────────────────────────
    // Warpgroup = which group of 128 threads am I in?
    // WG0: threads 0-127   → warpgroup_id = 0 → PRODUCER
    // WG1: threads 128-255 → warpgroup_id = 1 → CONSUMER
    int warpgroup_id = threadIdx.x / WARPGROUP_SIZE;
    int local_idx    = threadIdx.x % WARPGROUP_SIZE;  // 0..127 within WG

    // Record which role this thread has (for verification)
    // Only thread 0 of each WG records to avoid races
    if (local_idx == 0) {
        role_record[warpgroup_id] = warpgroup_id;
    }

    // ── Warp-specialized execution ────────────────────────────────
    // No single warp straddles this if-else boundary.
    // WG0's warps ALL take the if-branch.
    // WG1's warps ALL take the else-branch.
    // → Zero warp divergence.

    if (warpgroup_id == 0) {
        // ── PRODUCER path ──────────────────────────────────────
        // WG0: fill the shared memory buffer
        // Each of the 128 producer threads fills one slot
        producer_work(smem_buffer, local_idx, WARPGROUP_SIZE);

        // After filling, the producer has nothing else to do.
        // In a real kernel, it would loop to load the NEXT tile.
        // Here it just participates in the barrier below.

    } else {
        // ── CONSUMER path ──────────────────────────────────────
        // WG1: wait for producer data, then process it.
        // In a real kernel, we'd wait on an mbarrier here.
        // We fall through to __syncthreads() below.
    }

    // ── Barrier: ensure producer finished before consumer reads ──
    // In a real warp-specialized kernel, this is replaced by mbarrier.
    // __syncthreads() works here because it synchronizes ALL threads
    // in the block, including both WG0 and WG1.
    //
    // Timeline:
    //   WG0: fills smem_buffer ─────────────────────── │syncthreads│ idle
    //   WG1: idle (skips producer code) ──────────────── │syncthreads│ reads smem
    //                                                         ↑
    //                           Both WGs must reach here before either proceeds
    __syncthreads();

    // ── Consumer processes shared memory ─────────────────────────
    // Only WG1 threads do this work.
    // WG0 threads skip (warpgroup_id == 0 → not the else branch).
    if (warpgroup_id == 1) {
        float result = consumer_work(smem_buffer, local_idx);
        // Write result to global output
        // Consumer's local_idx 0..127 maps to output[0..127]
        output[local_idx] = result;
    }
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Warp Specialization Demo                             ║\n");
    printf("║  256 threads = WG0 (producer) + WG1 (consumer)       ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    const int OUTPUT_SIZE = WARPGROUP_SIZE;  // 128 outputs

    // ── Allocate device memory ────────────────────────────────────
    float* d_output;
    int*   d_roles;
    CHECK_CUDA(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_roles,  2 * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_roles,  0, 2 * sizeof(int)));

    // ── Launch: 1 block, 256 threads (2 warpgroups) ──────────────
    printf("Launch: 1 block × 256 threads\n");
    printf("  WG0 (threads   0-127): PRODUCER role\n");
    printf("  WG1 (threads 128-255): CONSUMER role\n\n");

    warp_specialized_kernel<<<1, 256>>>(d_output, d_roles);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ── Copy results back ─────────────────────────────────────────
    float h_output[OUTPUT_SIZE];
    int   h_roles[2];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_roles,  d_roles,  2 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // ── Verify role assignment ────────────────────────────────────
    printf("Role Assignment Check:\n");
    printf("  WG0 role recorded: %d (expected 0 = producer)\n", h_roles[0]);
    printf("  WG1 role recorded: %d (expected 1 = consumer)\n", h_roles[1]);
    bool roles_ok = (h_roles[0] == 0 && h_roles[1] == 1);
    printf("  Role assignment: %s\n\n", roles_ok ? "✓ CORRECT" : "✗ WRONG");

    // ── Verify consumer outputs ───────────────────────────────────
    printf("Consumer Output Verification:\n");
    printf("  Expected formula: output[i] = 2 × (3×i + 7) = 6i + 14\n");
    printf("  Checking all 128 outputs...\n\n");

    int errors = 0;
    // Print first 8 and last 4 for display
    printf("  %-8s %-12s %-12s %-8s\n", "Index", "Expected", "Got", "OK?");
    printf("  %s\n", "────────────────────────────────────────");

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float expected = 2.0f * (3.0f * i + 7.0f);  // = 6i + 14
        bool ok = (fabsf(h_output[i] - expected) < 1e-4f);
        if (!ok) errors++;

        // Only print first 5 and last 3
        if (i < 5 || i >= OUTPUT_SIZE - 3) {
            printf("  %-8d %-12.1f %-12.1f %s\n",
                   i, expected, h_output[i], ok ? "✓" : "✗ FAIL");
        } else if (i == 5) {
            printf("  ... (checking all %d, showing first 5 and last 3) ...\n",
                   OUTPUT_SIZE);
        }
    }

    // ── Summary ──────────────────────────────────────────────────
    printf("\n════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d output values correct\n", OUTPUT_SIZE - errors, OUTPUT_SIZE);
    printf("Role assignment: %s\n", roles_ok ? "CORRECT" : "INCORRECT");

    bool all_pass = (errors == 0 && roles_ok);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (all_pass) {
        printf("║  ✓ ALL TESTS PASSED                                  ║\n");
        printf("║                                                      ║\n");
        printf("║  KEY OBSERVATIONS:                                   ║\n");
        printf("║  • WG0 (threads 0-127) executed ONLY producer code   ║\n");
        printf("║  • WG1 (threads 128-255) executed ONLY consumer code  ║\n");
        printf("║  • No warp divergence within either warpgroup        ║\n");
        printf("║  • __syncthreads() ensured safe handoff              ║\n");
        printf("║  • In production: replace sync with mbarrier (§9)    ║\n");
    } else {
        printf("║  ✗ SOME TESTS FAILED — check output above            ║\n");
    }
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_roles));
    return all_pass ? 0 : 1;
}
```

---

## 5 — setmaxnreg — Dynamic Register Allocation

### 5.1 The Register Budget Problem

The H100's SM has 65,536 × 32-bit registers, divided among all active threads. A warpgroup running WGMMA needs ~200 registers per thread to hold the output tile accumulators. A producer warpgroup doing TMA loads needs only ~32 registers per thread.

If you launch a kernel naively at 200 registers/thread for everyone, you lose occupancy. If you launch at 32 registers/thread, the consumers can't hold their accumulators and spill to local memory (HBM) — catastrophically slow.

`setmaxnreg` solves this by letting warpgroups **dynamically resize their register allocation** at runtime.

```
setmaxnreg — What It Does and When to Use It
══════════════════════════════════════════════════════════════════════

  PTX instruction: setmaxnreg.inc.sync.aligned.u32 LIMIT;
                   setmaxnreg.dec.sync.aligned.u32 LIMIT;

  .inc = increase to LIMIT (consumer warpgroups call this)
  .dec = decrease to LIMIT (producer warpgroups call this)

  VALID LIMITS: Must be a multiple of 8. Common values:
    24  — absolute minimum
    32  — producer warps (TMA + barrier management only)
    40  — small compute tasks
    128 — moderate accumulator
    160 — typical consumer warpgroup
    232 — maximum practical (leaving room for control logic)
    255 — hardware maximum

  ─── The Timeline ─────────────────────────────────────────────────

  Kernel launch at, say, 64 registers/thread (fits 1024 threads/SM):

  WG0 (producer) execution:
    [setup]
    setmaxnreg.dec.sync.aligned.u32 32;  ← "I only need 32 regs"
    [producer loop: TMA loads, barrier management]
    [uses only 32 registers → SM can now schedule more warps here]

  WG1 (consumer) execution:
    [setup]
    setmaxnreg.inc.sync.aligned.u32 232; ← "I need 232 regs for accumulators"
    [consumer loop: WGMMA, holds large accumulator in 232 registers]
    [SM grants 232 regs; fewer consumer threads can fit per SM, but that's ok]

  ─── What Actually Happens in Hardware ───────────────────────────

  Before setmaxnreg:
    All threads allocated launch-time register count (e.g., 64)
    Register file: 65536 regs / 64 regs/thread = 1024 threads active

  After WG0 calls setmaxnreg.dec(32):
    WG0 frees (64-32) × 128 = 4096 registers back to the pool
    These freed registers can be given to other warps

  After WG1 calls setmaxnreg.inc(232):
    WG1 requests (232-64) × 128 = 21504 more registers from the pool
    If available, WG1 gets 232 registers per thread
    If not available, WG1 stalls (rare — WG0 already freed some)

  ─── The Right Pattern ───────────────────────────────────────────
  1. Launch kernel at a LOW register count (e.g., 64)
     → More threads can be scheduled initially → better memory latency hiding
  2. Producer warpgroup calls setmaxnreg.dec early in its code
     → Frees registers for consumers to claim
  3. Consumer warpgroup calls setmaxnreg.inc before its WGMMA loop
     → Claims the freed registers for accumulators

  ─── C++ Interface ───────────────────────────────────────────────
  // CUDA 12.3+ provides intrinsics:
  __syncwarp();
  // Decrease (for producers):
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(32));
  // Increase (for consumers):
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(232));

  // The "n" constraint = compile-time constant (required by hardware)
  // The .sync means all threads in the warp must reach this point
  // The .aligned means warpgroup must reach it together (128 threads)
```

---

### ★ Section 5 — Runnable Code: setmaxnreg Demo

**Manual Worked Example:**

We launch 256 threads. WG0 decreases to 32 registers (producer), WG1 increases to 128 registers (consumer). We verify that:
1. The PTX executes without error
2. After setmaxnreg, threads can still compute correctly
3. The register counts are (conceptually) different between warpgroups

```
Register allocation timeline:
  Launch:  all 256 threads allocated 64 registers each
           SM holds: 65536 / 64 = 1024 threads per SM (high occupancy)

  After WG0 setmaxnreg.dec(32):
    WG0 has 32 regs/thread = 128 × 32 = 4096 regs total
    Released: 128 × (64-32) = 4096 regs back to pool

  After WG1 setmaxnreg.inc(128):
    WG1 wants 128 regs/thread = 128 × 128 = 16384 regs total
    Requests: 128 × (128-64) = 8192 additional regs from pool
    Pool had 4096 freed from WG0 + remaining allocation → available
```

```
═══════════════════════════════════════════════════════
  FILE:    ex03_setmaxnreg.cu
  COMPILE: nvcc -std=c++17 -arch=sm_90a -O2 ex03_setmaxnreg.cu -o ex03_maxreg
           NOTE: setmaxnreg is a Hopper (sm_90a+) instruction.
                 Will not compile on sm_80 or earlier.
  RUN:     ./ex03_maxreg
  EXPECT:  Both warpgroups produce correct results after setmaxnreg,
           demonstrating that register resize does not corrupt state.
═══════════════════════════════════════════════════════
```

```cuda
// ═══════════════════════════════════════════════════════════════════
// ex03_setmaxnreg.cu
//
// PURPOSE: Demonstrate setmaxnreg — dynamic register resizing.
//
// DESIGN:
//   WG0 (threads 0-127):   calls setmaxnreg.dec(32) → producer registers
//   WG1 (threads 128-255): calls setmaxnreg.inc(128) → consumer registers
//
//   After resizing, both warpgroups perform arithmetic to prove
//   that register state is preserved through the resize operation.
//
// IMPORTANT: setmaxnreg MUST be aligned (all threads in warpgroup
// must execute it simultaneously, hence .aligned variant).
//
// REQUIRES: sm_90a (Hopper) or sm_100a (Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

constexpr int WARPGROUP_SIZE = 128;
constexpr int PRODUCER_REGS  = 32;   // minimum for producer (TMA + barriers)
constexpr int CONSUMER_REGS  = 128;  // moderate consumer register budget

// ── Kernel ────────────────────────────────────────────────────────
// Tests that setmaxnreg works correctly and doesn't corrupt computation.
//
// Protocol:
//   1. All threads compute an initial value (before setmaxnreg)
//   2. setmaxnreg changes register allocation
//   3. All threads perform MORE arithmetic (uses same registers!)
//   4. We verify that the final result is correct
//
// If setmaxnreg corrupted registers, the second computation would be wrong.
__global__ void test_setmaxnreg(float* output_wg0, float* output_wg1) {
    int warpgroup_id = threadIdx.x / WARPGROUP_SIZE;
    int local_idx    = threadIdx.x % WARPGROUP_SIZE;

    // ── Step 1: Compute initial value (before any register resize) ─
    // Both warpgroups do this. The value lives in a register.
    float accumulator = (float)(local_idx * 2 + 1);  // 1, 3, 5, 7, ...
    float factor      = (float)(warpgroup_id + 1);    // WG0: 1.0, WG1: 2.0

    if (warpgroup_id == 0) {
        // ── PRODUCER warpgroup: decrease registers ────────────────
        // setmaxnreg.dec: "I am releasing registers down to PRODUCER_REGS"
        // The .sync means: this thread can't issue instructions after
        //                  this point until ALL threads in the warpgroup reach it.
        // The .aligned means: every thread in the warp group must
        //                     execute this together.
        asm volatile(
            "setmaxnreg.dec.sync.aligned.u32 %0;"
            :: "n"(PRODUCER_REGS)
        );
        // After this point: WG0 has PRODUCER_REGS registers available.
        // The register 'accumulator' and 'factor' STILL EXIST because
        // the compiler arranged them to fit within PRODUCER_REGS.
        // (If they didn't fit, the compiler would spill — check with --ptxas-options=-v)

    } else {
        // ── CONSUMER warpgroup: increase registers ─────────────────
        // setmaxnreg.inc: "I am claiming more registers up to CONSUMER_REGS"
        asm volatile(
            "setmaxnreg.inc.sync.aligned.u32 %0;"
            :: "n"(CONSUMER_REGS)
        );
        // After this point: WG1 has CONSUMER_REGS registers available.
        // The existing registers 'accumulator' and 'factor' are still valid.
        // WG1 now has room for more intermediate results.
    }

    // ── Step 2: More computation AFTER register resize ────────────
    // This tests that setmaxnreg did not corrupt the existing register state.
    // If registers were corrupted, this computation would produce wrong results.
    //
    // We do a small computation that uses the pre-resize values:
    //   result = accumulator * factor + local_idx * 0.5
    //
    // WG0: accumulator = 2*idx+1, factor = 1.0 → result = 2*idx+1 + 0.5*idx = 2.5*idx+1
    // WG1: accumulator = 2*idx+1, factor = 2.0 → result = 2*(2*idx+1) + 0.5*idx = 4.5*idx+2
    float result = accumulator * factor + (float)local_idx * 0.5f;

    // Do a few more ops to stress-test register integrity
    result = result * 1.0f + 0.0f;  // identity: should not change result
    result = result + 0.0f;

    // ── Step 3: Write results ──────────────────────────────────────
    if (warpgroup_id == 0) {
        output_wg0[local_idx] = result;
    } else {
        output_wg1[local_idx] = result;
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  setmaxnreg Demo — Dynamic Register Resizing         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Check for Hopper
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: setmaxnreg requires Hopper (SM 9.0+) or Blackwell.\n");
        return 1;
    }

    printf("Register Plan:\n");
    printf("  WG0 (threads   0-127): setmaxnreg.dec → %d regs (producer)\n",
           PRODUCER_REGS);
    printf("  WG1 (threads 128-255): setmaxnreg.inc → %d regs (consumer)\n\n",
           CONSUMER_REGS);

    float *d_wg0, *d_wg1;
    CHECK_CUDA(cudaMalloc(&d_wg0, WARPGROUP_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wg1, WARPGROUP_SIZE * sizeof(float)));

    test_setmaxnreg<<<1, 256>>>(d_wg0, d_wg1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_wg0[WARPGROUP_SIZE], h_wg1[WARPGROUP_SIZE];
    CHECK_CUDA(cudaMemcpy(h_wg0, d_wg0, WARPGROUP_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wg1, d_wg1, WARPGROUP_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    // Verify WG0 results: result = (2*idx+1)*1.0 + idx*0.5 = 2.5*idx + 1
    printf("WG0 (PRODUCER, %d regs after resize):\n", PRODUCER_REGS);
    int errors_wg0 = 0, errors_wg1 = 0;
    for (int i = 0; i < WARPGROUP_SIZE; i++) {
        float expected = (2.0f * i + 1.0f) * 1.0f + (float)i * 0.5f;
        if (fabsf(h_wg0[i] - expected) > 1e-4f) {
            errors_wg0++;
            if (errors_wg0 <= 3)
                printf("  FAIL [%d]: got %.4f expected %.4f\n", i, h_wg0[i], expected);
        }
    }
    if (errors_wg0 == 0) {
        printf("  ✓ All %d values correct (registers NOT corrupted by setmaxnreg)\n",
               WARPGROUP_SIZE);
        printf("  Sample: output[0]=%.1f (expect 1.0), output[1]=%.1f (expect 3.5)\n",
               h_wg0[0], h_wg0[1]);
    }

    // Verify WG1 results: result = (2*idx+1)*2.0 + idx*0.5 = 4.5*idx + 2
    printf("\nWG1 (CONSUMER, %d regs after resize):\n", CONSUMER_REGS);
    for (int i = 0; i < WARPGROUP_SIZE; i++) {
        float expected = (2.0f * i + 1.0f) * 2.0f + (float)i * 0.5f;
        if (fabsf(h_wg1[i] - expected) > 1e-4f) {
            errors_wg1++;
            if (errors_wg1 <= 3)
                printf("  FAIL [%d]: got %.4f expected %.4f\n", i, h_wg1[i], expected);
        }
    }
    if (errors_wg1 == 0) {
        printf("  ✓ All %d values correct (registers NOT corrupted by setmaxnreg)\n",
               WARPGROUP_SIZE);
        printf("  Sample: output[0]=%.1f (expect 2.0), output[1]=%.1f (expect 6.5)\n",
               h_wg1[0], h_wg1[1]);
    }

    bool all_pass = (errors_wg0 == 0 && errors_wg1 == 0);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (all_pass) {
        printf("║  ✓ ALL TESTS PASSED                                  ║\n");
        printf("║  setmaxnreg correctly resized registers without      ║\n");
        printf("║  corrupting existing register state.                 ║\n");
    } else {
        printf("║  ✗ FAILURES: WG0=%d errors, WG1=%d errors            ║\n",
               errors_wg0, errors_wg1);
    }
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_wg0)); CHECK_CUDA(cudaFree(d_wg1));
    return all_pass ? 0 : 1;
}
```

---

## 6 — Warp Group Count Selection

### 6.1 Choosing How Many Consumer Warpgroups

The number of consumer warpgroups determines how large a tile each consumer accumulates, and therefore the kernel's overall arithmetic intensity. This is one of the most important tuning decisions in a GEMM kernel.

```
Warp Group Count Selection — Decision Tree
══════════════════════════════════════════════════════════════════════

  Register Budget:
    Total SM registers: 65,536 × 32-bit
    Producer warpgroup (1×WG): 128 threads × 32 regs = 4,096 regs
    Remaining for consumers: 65,536 - 4,096 = 61,440 regs

  Per consumer thread: 232 registers (target for large accumulators)
  Threads per consumer WG: 128
  Regs per consumer WG: 128 × 232 = 29,696 regs

  Consumer WG choices:
  ─────────────────────────────────────────────────────────────────
    1 Consumer WG:
      Total regs: 4096 + 29696 = 33792 (uses ~52% of SM regs)
      Effective MMA threads: 128
      Output tile held in regs: ~128×64 floats  (typical)
      Use when: tile is 64×128 or 128×128, standard GEMM

    2 Consumer WGs (Ping-Pong or Cooperative):
      Total regs: 4096 + 2×29696 = 63488 (uses ~97% of SM regs)
      Effective MMA threads: 256
      Output tile held in regs: ~256×64 floats  (doubled!)
      Use when: tile is 128×128 or 256×128, want epilogue overlap

    3 Consumer WGs (rare):
      Total regs: 4096 + 3×29696 = exceeds SM capacity!
      → Not feasible at 232 regs/thread
      → Must reduce to ~160 regs/consumer thread for 3 WGs

  ─── Register Budget vs Tile Size ────────────────────────────────

  WGMMA accumulates into registers. For FP16→FP32 WGMMA with tile MxN:
    Accumulator registers = M × N / (threads_per_WG × REGS_PER_ELEMENT)
    For M=64, N=64: 64×64 floats / 128 threads = 32 floats/thread = 32 regs
    For M=64, N=128: 64×128 floats / 128 threads = 64 floats = 64 regs
    For M=64, N=256: 128 regs  (2 WGs needed: 64 regs each)
    For M=128, N=128: 128 regs (2 WGs cooperative: 64 regs each)

  ─── The Practical Decision ──────────────────────────────────────

  Use 1 consumer WG when:
    ├── Tile ≤ 128×128 (accumulator fits in 1 WG's budget)
    ├── K-loop is short (epilogue not a bottleneck)
    └── Simple kernel with few epilogue ops

  Use 2 consumer WGs (Ping-Pong) when:
    ├── Tile = 128×128, long K-loop
    ├── Epilogue is expensive (many fused ops, FP8 scaling, etc.)
    └── Want WGMMA and epilogue to overlap (Section 14)

  Use 2 consumer WGs (Cooperative) when:
    ├── Tile > 128×128 (too big for 1 WG, but no epilogue overlap needed)
    ├── K-loop dominates runtime (epilogue is fast)
    └── Prefer fewer CTAs per tile with larger effective WGMMA width

  Use Cluster of CTAs when:
    └── Tile > 256×128 (even 2 cooperative WGs can't hold accumulators)
        Multiple CTAs share computation via DSMEM (Lesson 2)
```

---

## 7 — Pipelining

### 7.1 Why Pipelining Is Mandatory

The H100's HBM3 has ~500 cycle latency for individual memory accesses. WGMMA executes in ~30-60 cycles. Without pipelining, every time the consumer needs new tile data, it waits 500 cycles of doing nothing. This is a 10-17× utilization gap.

Pipelining eliminates this idle time by overlapping the DATA MOVEMENT for tile N+1 with the COMPUTATION of tile N. The consumer is always computing while the producer is always loading.

```
Pipelining — What It Means Concretely
══════════════════════════════════════════════════════════════════════

  Timeline WITHOUT pipelining:
  ─────────────────────────────────────────────────────────────────
  Cycle:  0    100  200  300  400  500  600  700  800  900  1000
          ├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
  Loads:  ├─────────────────Load T0──────────────────┤
  Math:                                              ├─MMA T0─┤
  Loads:                                                      ├────Load T1──────
  Math:   (idle during load T0)

  Utilization: ~50 cycles compute / ~550 cycles total ≈ 9%

  Timeline WITH 2-stage pipelining:
  ─────────────────────────────────────────────────────────────────
  Cycle:  0    100  200  300  400  500  600  700  800  900  1000
          ├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
  Loads:  ├──Load T0──────────────────┤
          │                      ├──Load T1──────────────────┤
          │                                             ├──Load T2──
  Math:                               ├─MMA T0─┤├─MMA T1─┤├─MMA T2─
                                      ↑ overlapped with Load T1!

  Utilization: >95% (loads and math overlap continuously)

  Timeline WITH 4-stage pipelining (more overlap, better hiding):
  ─────────────────────────────────────────────────────────────────
  Stage 0: [Load T0]
  Stage 1: [Load T1][MMA T0]
  Stage 2: [Load T2][MMA T1]
  Stage 3: [Load T3][MMA T2]
  Stage 4:          [MMA T3]  (drain)
  ...and so on, steady state: always one load + one MMA per step

  The key insight: with N pipeline stages, we can hide up to N×(load latency)
  of memory latency. With 4 stages and 500-cycle loads, we hide 2000 cycles!

  ─── The Mathematical Condition for Full Overlap ──────────────────
  For complete hiding of load latency:
    num_stages × stage_time_math > load_latency
    num_stages × MMA_time > HBM_latency
    num_stages × ~50 cycles > ~500 cycles
    num_stages > 10

  Wait — in practice, TMA hides the HBM latency internally (it's a
  DMA engine, not thread-synchronous). What we actually need:
    num_stages × MMA_time > TMA_issue_to_completion_latency
    In practice: 3-4 stages is sufficient for H100 with TMA.
```

### 7.2 What the Pipeline Stages Physically Are

Each "stage" in the pipeline corresponds to a physical allocation of shared memory. With 4 stages, you allocate 4 copies of the A-tile and B-tile in shared memory. This allows 4 tiles to be "in flight" simultaneously: some being loaded by TMA, others being consumed by WGMMA.

```
Pipeline Stages — Physical SMEM Layout
══════════════════════════════════════════════════════════════════════

  SMEM layout with 4 pipeline stages (stages = circular buffer slots):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Shared Memory (228 KB available)                               │
  │                                                                 │
  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
  │  │  Stage 0   │  │  Stage 1   │  │  Stage 2   │  │  Stage 3   │ │
  │  │            │  │            │  │            │  │            │ │
  │  │ A tile 0   │  │ A tile 1   │  │ A tile 2   │  │ A tile 3   │ │
  │  │ (32KB)     │  │ (32KB)     │  │ (32KB)     │  │ (32KB)     │ │
  │  │            │  │            │  │            │  │            │ │
  │  │ B tile 0   │  │ B tile 1   │  │ B tile 2   │  │ B tile 3   │ │
  │  │ (32KB)     │  │ (32KB)     │  │ (32KB)     │  │ (32KB)     │ │
  │  │            │  │            │  │            │  │            │ │
  │  │ mbarrier   │  │ mbarrier   │  │ mbarrier   │  │ mbarrier   │ │
  │  │ full + empty│  │ full + empty│  │ full + empty│  │ full + empty│
  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
  │                                                                 │
  │  Total for 4 stages: 4 × (32+32) KB = 256 KB (tight fit!)      │
  │  Epilogue buffer: remaining space (for bias, output, etc.)      │
  └─────────────────────────────────────────────────────────────────┘

  Stage lifecycle (think of it as a rental unit):
    EMPTY: available for producer to load into
    LOADING: producer is filling via TMA (async)
    FULL: ready for consumer to use
    IN_USE: consumer is doing WGMMA from this stage
    (→ back to EMPTY when consumer signals done)

  The number of stages (typically 3 or 4) must satisfy:
    num_stages × (A_tile_bytes + B_tile_bytes) ≤ available_SMEM

  For H100 with 228KB SMEM and 32×32KB tiles:
    num_stages ≤ 228KB / 64KB = 3.5 → use 3 stages maximum
    (often 4 with 16KB tiles for FP8 or smaller K-dimension tiles)
```

---

## 8 — Circular Buffering and Multi-Stage Pipelines

### 8.1 The Circular Buffer as a Ring

With N pipeline stages in shared memory, the producer and consumer work through the stages in a circular (ring) fashion. The producer writes to stage `i % N`, the consumer reads from stage `(i - depth) % N`, where `depth` is how many stages ahead the producer stays.

```
Circular Buffer — Ring Operation
══════════════════════════════════════════════════════════════════════

  N = 4 stages. Producer stays "depth=2" steps ahead.
  Read pointer = what consumer is currently reading from.
  Write pointer = what producer is currently loading into.

  Iteration 0: Write=0, Read=N/A  (producer fills stage 0)
  Iteration 1: Write=1, Read=N/A  (producer fills stage 1)
  Iteration 2: Write=2, Read=0    (producer fills 2, consumer reads 0)
  Iteration 3: Write=3, Read=1    (producer fills 3, consumer reads 1)
  Iteration 4: Write=0, Read=2    (producer wraps to 0, consumer reads 2)
  Iteration 5: Write=1, Read=3    (producer wraps to 1, consumer reads 3)
  Iteration 6: Write=2, Read=0    (producer wraps to 2, consumer reads 0)
  ...

  Stage ownership at iteration 5 (write=1, read=3):
  ┌─────────┬─────────┬─────────┬─────────┐
  │ Stage 0 │ Stage 1 │ Stage 2 │ Stage 3 │
  │         │         │         │         │
  │ FULL    │ LOADING │ FULL    │ IN_USE  │
  │ (next   │ (prod   │ (next   │ (cons   │
  │ for     │ writing │ in      │ reading │
  │ consumer│ here)   │ queue)  │ here)   │
  └─────────┴─────────┴─────────┴─────────┘
          ↑                           ↑
     Write pointer                Read pointer
     (write % 4 = 1)              (read % 4 = 3)

  ─── The "Prologue" and "Drain" Phases ────────────────────────────
  The pipeline has three phases:

  PROLOGUE (fill the pipeline):
    Producer fills stages 0, 1, ... (depth-1)
    Consumer has nothing to read yet
    Duration: `depth` iterations

  STEADY STATE (full overlap):
    Producer fills stage (i % N)
    Consumer reads stage ((i - depth) % N)
    Both run simultaneously — full overlap

  DRAIN (empty the pipeline):
    Producer has no more tiles to fill (K-loop over)
    Consumer still has `depth` tiles left to process
    Duration: `depth` iterations

  ─── Why Multiple Stages Are NECESSARY ────────────────────────────
  With 1 stage: Producer fills → waits for consumer → consumer reads
  → waits for producer → NO overlap. Just serialized execution.

  With 2 stages:
    Producer fills stage 0.
    Producer fills stage 1 (while consumer reads stage 0!)
    Consumer reads stage 0.
    Producer fills stage 0 (while consumer reads stage 1!)
    Consumer reads stage 1.
    → Partial overlap. Producer may still stall if consumer is slow.

  With 4 stages:
    Producer can be 3 tiles ahead of the consumer at all times.
    As long as TMA latency < 3 × MMA_time, producer never stalls.
    This is the typical operating condition on H100.
```

---

### ★ Section 8 — Runnable Code: Circular Buffer Simulation

**Manual Worked Example:**

We simulate a circular buffer with N=4 stages, K=8 k-tiles (iterations), and depth=2. We trace which stage the producer and consumer use at each step.

```
Expected stage usage trace:
  Iteration 0: Producer writes stage 0. Consumer: none yet.
  Iteration 1: Producer writes stage 1. Consumer: none yet (prologue).
  Iteration 2: Producer writes stage 2. Consumer reads stage 0.
  Iteration 3: Producer writes stage 3. Consumer reads stage 1.
  Iteration 4: Producer writes stage 0. Consumer reads stage 2.
  Iteration 5: Producer writes stage 1. Consumer reads stage 3.
  Iteration 6: Producer writes stage 2. Consumer reads stage 0.
  Iteration 7: Producer writes stage 3. Consumer reads stage 1.
  Drain  0:    (no more writes).        Consumer reads stage 2.
  Drain  1:    (no more writes).        Consumer reads stage 3.
```

```
═══════════════════════════════════════════════════════
  FILE:    ex04_circular_buffer.cu
  COMPILE: nvcc -std=c++17 -arch=sm_90a -O2 ex04_circular_buffer.cu -o ex04_cbuf
           (Also works on sm_80 — no Hopper-specific features here)
  RUN:     ./ex04_circular_buffer
  EXPECT:  Correct circular stage assignments at every iteration,
           verified against the manual trace above.
           Buffer correctly accumulates values across all K iterations.
═══════════════════════════════════════════════════════
```

```cuda
// ═══════════════════════════════════════════════════════════════════
// ex04_circular_buffer.cu
//
// PURPOSE: Simulate a circular pipeline buffer on GPU.
//   Demonstrates the ring indexing, prologue, steady-state, and drain
//   phases without requiring actual TMA or WGMMA hardware.
//
// SIMULATION:
//   "Producer" threads: write k*10 + stage_index into the SMEM ring slot
//   "Consumer" threads: read from ring slot, accumulate into output
//
//   With N=4 stages and K=8 k-tiles, we verify:
//   1. Each consumer reads are the correct values (written by producer)
//   2. The ring wrapping works correctly at the stage-4 boundary
//   3. The prologue fills exactly `depth` stages before consuming
//
// WORKS ON: sm_80+ (ring buffer logic is architecture-independent)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

constexpr int N_STAGES   = 4;    // number of pipeline stages (ring slots)
constexpr int PIPELINE_DEPTH = 2;  // how far ahead producer stays
constexpr int K_TILES    = 8;    // total k-tile iterations
constexpr int TILE_SIZE  = 64;   // floats per tile (simulated)

// Sentinel value for unwritten slots
constexpr float UNWRITTEN = -999.0f;

// ── Circular buffer pipeline kernel ──────────────────────────────
// This kernel simulates the producer-consumer ring buffer pattern
// using a sequential loop (single thread) for clarity.
//
// In a real kernel:
//   - Producer runs in a separate warpgroup (in parallel)
//   - Stages have mbarrier full/empty barriers (not simulated here)
//   - Producer uses TMA, consumer uses WGMMA
// Here, we use a single-thread sequential simulation to verify
// the index arithmetic and data flow logic.
__global__ void simulate_pipeline(
    float* trace_prod_stage,   // which stage producer writes at each iter
    float* trace_cons_stage,   // which stage consumer reads at each iter
    float* final_output,       // accumulated sum over all K tiles
    int*   errors_out          // error count
)
{
    // Only thread 0 runs this simulation
    if (threadIdx.x != 0) return;

    // ── Circular buffer in shared memory ──────────────────────────
    // Each stage holds one tile (TILE_SIZE floats).
    __shared__ float ring[N_STAGES][TILE_SIZE];
    __shared__ bool stage_full[N_STAGES];  // is stage ready for consumer?

    // Initialize
    for (int s = 0; s < N_STAGES; s++) {
        stage_full[s] = false;
        for (int j = 0; j < TILE_SIZE; j++) ring[s][j] = UNWRITTEN;
    }

    float output[TILE_SIZE] = {};  // consumer accumulator
    int errors = 0;

    // ── PHASE 1: PROLOGUE ─────────────────────────────────────────
    // Producer fills the first `PIPELINE_DEPTH` stages.
    // Consumer has nothing to read yet.
    printf("[Prologue]: filling first %d stages\n", PIPELINE_DEPTH);
    for (int k = 0; k < PIPELINE_DEPTH && k < K_TILES; k++) {
        int stage = k % N_STAGES;

        // Producer writes: value = k * 10.0 + j * 0.01 (unique per k-tile + element)
        for (int j = 0; j < TILE_SIZE; j++) {
            ring[stage][j] = k * 10.0f + j * 0.01f;
        }
        stage_full[stage] = true;

        trace_prod_stage[k] = stage;
        trace_cons_stage[k] = -1.0f;  // no consumer yet

        printf("  k=%d: Producer writes stage %d  |  Consumer: waiting\n", k, stage);
    }

    // ── PHASE 2: STEADY STATE ─────────────────────────────────────
    // Producer fills stage k, consumer reads stage k-DEPTH.
    printf("\n[Steady State]: producer and consumer overlap\n");
    for (int k = PIPELINE_DEPTH; k < K_TILES; k++) {
        int prod_stage = k % N_STAGES;
        int cons_stage = (k - PIPELINE_DEPTH) % N_STAGES;

        // Consumer reads from cons_stage (verifying stage_full)
        if (!stage_full[cons_stage]) {
            printf("  ERROR at k=%d: consumer's stage %d not full!\n", k, cons_stage);
            errors++;
        } else {
            // Accumulate into output
            for (int j = 0; j < TILE_SIZE; j++) {
                output[j] += ring[cons_stage][j];
            }
            stage_full[cons_stage] = false;  // release stage back to producer
        }

        // Producer writes next tile into prod_stage
        for (int j = 0; j < TILE_SIZE; j++) {
            ring[prod_stage][j] = k * 10.0f + j * 0.01f;
        }
        stage_full[prod_stage] = true;

        trace_prod_stage[k] = prod_stage;
        trace_cons_stage[k] = cons_stage;

        printf("  k=%d: Producer writes stage %d  |  Consumer reads stage %d\n",
               k, prod_stage, cons_stage);
    }

    // ── PHASE 3: DRAIN ────────────────────────────────────────────
    // No more production. Consumer drains remaining DEPTH stages.
    printf("\n[Drain]: flushing %d remaining stages\n", PIPELINE_DEPTH);
    for (int d = 0; d < PIPELINE_DEPTH; d++) {
        int cons_k = K_TILES - PIPELINE_DEPTH + d;
        int cons_stage = cons_k % N_STAGES;

        if (!stage_full[cons_stage]) {
            printf("  ERROR in drain d=%d: stage %d not full!\n", d, cons_stage);
            errors++;
        } else {
            for (int j = 0; j < TILE_SIZE; j++) {
                output[j] += ring[cons_stage][j];
            }
            stage_full[cons_stage] = false;
        }

        trace_prod_stage[K_TILES + d] = -1.0f;  // no more production
        trace_cons_stage[K_TILES + d] = cons_stage;

        printf("  drain %d: Consumer reads stage %d  |  Producer: done\n", d, cons_stage);
    }

    // Write final output (just element 0 and element 1 for verification)
    for (int j = 0; j < TILE_SIZE; j++) final_output[j] = output[j];
    *errors_out = errors;
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Circular Pipeline Buffer Simulation                  ║\n");
    printf("║  N_STAGES=%d, DEPTH=%d, K_TILES=%d                    ║\n",
           N_STAGES, PIPELINE_DEPTH, K_TILES);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    const int TOTAL_ITERS = K_TILES + PIPELINE_DEPTH;  // prologue + steady + drain

    float *d_prod_trace, *d_cons_trace, *d_output;
    int*   d_errors;
    CHECK_CUDA(cudaMalloc(&d_prod_trace, TOTAL_ITERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cons_trace, TOTAL_ITERS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output,     TILE_SIZE   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_errors,     sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));

    simulate_pipeline<<<1, 64>>>(d_prod_trace, d_cons_trace, d_output, d_errors);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_prod[TOTAL_ITERS], h_cons[TOTAL_ITERS], h_out[TILE_SIZE];
    int   h_errors;
    CHECK_CUDA(cudaMemcpy(h_prod,   d_prod_trace, TOTAL_ITERS*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cons,   d_cons_trace, TOTAL_ITERS*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out,    d_output,     TILE_SIZE  *sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_errors,d_errors,     sizeof(int),               cudaMemcpyDeviceToHost));

    // ── Verify stage assignments ──────────────────────────────────
    printf("\n════════════════════════════════════════════════════════\n");
    printf("Stage Assignment Verification:\n");
    printf("  Expected producer: 0,1,2,3,0,1,2,3,-1,-1 (then drain)\n");
    printf("  Expected consumer: -1,-1,0,1,2,3,0,1,2,3\n\n");

    // Expected producer stages: k % N_STAGES for k=0..K_TILES-1, then -1 for drain
    // Expected consumer stages: -1 for prologue, then (k-DEPTH)%N for rest
    int verify_errors = 0;
    printf("  %4s  %6s  %6s\n", "Iter", "Prod", "Cons");
    printf("  %s\n", "────────────────────");
    for (int i = 0; i < TOTAL_ITERS; i++) {
        float exp_prod = (i < K_TILES) ? (float)(i % N_STAGES) : -1.0f;
        float exp_cons = (i < PIPELINE_DEPTH) ? -1.0f :
                         (float)((i - PIPELINE_DEPTH) % N_STAGES);
        bool ok = (fabsf(h_prod[i] - exp_prod) < 0.1f &&
                   fabsf(h_cons[i] - exp_cons) < 0.1f);
        if (!ok) verify_errors++;
        printf("  %4d  %6.0f  %6.0f %s\n", i, h_prod[i], h_cons[i],
               ok ? "✓" : "✗ FAIL");
    }

    // ── Verify accumulated output ─────────────────────────────────
    // Element j accumulated over k=0..7: sum of k*10 + j*0.01 for k=0..7
    // = (0+10+20+30+40+50+60+70) + j*0.01*8 = 280 + 0.08*j
    printf("\nOutput Verification (element 0 and element 10):\n");
    float expected_0  = 280.0f + 0.0f;   // j=0: sum = 280
    float expected_10 = 280.0f + 0.08f * 10;  // j=10: sum = 281
    printf("  output[0]  = %.3f (expected %.3f) %s\n",
           h_out[0], expected_0,
           fabsf(h_out[0] - expected_0) < 0.01f ? "✓" : "✗ FAIL");
    printf("  output[10] = %.3f (expected %.3f) %s\n",
           h_out[10], expected_10,
           fabsf(h_out[10] - expected_10) < 0.01f ? "✓" : "✗ FAIL");

    bool all_ok = (verify_errors == 0 && h_errors == 0 &&
                   fabsf(h_out[0] - expected_0) < 0.01f);

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  Stage assignment errors: %d                           ║\n", verify_errors);
    printf("║  Runtime barrier errors:  %d                           ║\n", h_errors);
    if (all_ok)
        printf("║  ✓ ALL TESTS PASSED — circular buffer correct        ║\n");
    else
        printf("║  ✗ SOME FAILURES — check output above               ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_prod_trace)); CHECK_CUDA(cudaFree(d_cons_trace));
    CHECK_CUDA(cudaFree(d_output)); CHECK_CUDA(cudaFree(d_errors));
    return all_ok ? 0 : 1;
}
```

---

> **Next — Chapter 9: Pipelines, Epilogues & Persistent Scheduling**
>
> This chapter's numbering runs straight on into the next one: Chapter 9 picks up
> at §9 with the two-barrier handshake, then covers ABA ambiguity and phase bits,
> cooperative versus ping-pong pipeline architectures, the six-stage epilogue
> fusion lifecycle, persistent scheduling, and rasterisation for L2 locality.
