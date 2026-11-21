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
//   AI ≈ 85.3 FLOP/Byte → well above FP32 ridge (~20 FLOP/Byte)
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
        printf("  → AI=%.1f >> 20 → HIGH-AI: inspect issue/shared-memory/dependency limits\n\n", ai);

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
        printf("  → AI selects the compute roof; measured bottleneck is not established (bandwidth not saturated)\n");

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
