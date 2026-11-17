// ═══════════════════════════════════════════════════════════════════
// ex01_wgmma_pipeline.cu
//
// PURPOSE: Demonstrate wgmma commit/wait groups with stmatrix epilogue.
//   Shows:
//     1. wgmma.commit_group to batch math operations
//     2. wgmma.wait_group<1> to allow overlap (1 group still running)
//     3. wgmma.wait_group<0> to drain all groups
//     4. wgmma.fence before and after wgmma sequences
//     5. stmatrix for writing FP16 accumulators back to SMEM
//     6. Verification of both tile results
//
// Uses FP16 accumulators (not FP32) to enable direct stmatrix without conversion.
// REQUIRES: Hopper sm_90a (H100/H200); not a Blackwell kernel
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} } while(0)

static constexpr int M = 64;
static constexpr int N = 64;
static constexpr int K = 16;
// FP16 accumulator: (M×N)/128 = 32 FP16 values = 16 packed .b32 registers
static constexpr int D_REGS_F16 = (M * N) / 128 / 2;  // 16 .b32 registers

// ─────────────────────────────────────────────────────────────────
// Build WGMMA descriptor (simplified K-major FP16, 128B swizzle)
// ─────────────────────────────────────────────────────────────────
__device__ uint64_t make_desc(const void* smem_ptr,
                               uint32_t sbo_bytes, int swizzle_mode = 0) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)(8ULL & 0x3FFF) << 16;               // LBO=128B for no-swizzle core matrices
    d |= (uint64_t)((sbo_bytes/16) & 0x3FFF) << 32;     // SBO
    d |= (uint64_t)(0ULL) << 49;                          // base offset = 0
    d |= (uint64_t)((uint64_t)swizzle_mode) << 62;       // swizzle mode bits
    return d;
}

// ─────────────────────────────────────────────────────────────────
// KERNEL: demonstrates commit/wait pipeline + stmatrix epilogue
// Two tiles A and B are multiplied; results verified in SMEM.
// ─────────────────────────────────────────────────────────────────
__global__ void wgmma_pipeline_kernel(
    const __half* A_glob,
    const __half* B_glob,
    __half* out_A,
    __half* out_B,
    uint32_t* pass_count)
{
    // ── Shared memory (two A tiles + two B tiles for two output tiles) ──
    // Tile A's A matrix and B matrix, then Tile B's A and B matrices.
    __shared__ __align__(1024) __half smem_Aa[M][K];  // A input for tile A
    __shared__ __align__(1024) __half smem_Ba[K][N];  // B input for tile A
    __shared__ __align__(1024) __half smem_Ab[M][K];  // A input for tile B
    __shared__ __align__(1024) __half smem_Bb[K][N];  // B input for tile B
    // Output staging buffer (reused for both tiles)
    __shared__ __align__(1024) __half smem_out[M][N];

    int tid = threadIdx.x;

    // ── Load input tiles from global memory (simple thread-driven) ───
    for (int i = tid; i < M*K; i += blockDim.x) {
        int r = i/K, c = i%K;
        reinterpret_cast<__half*>(smem_Aa)[(r/8)*128+(c/8)*64+(r%8)*8+c%8] = A_glob[r*K+c];
        reinterpret_cast<__half*>(smem_Ab)[(r/8)*128+(c/8)*64+(r%8)*8+c%8] = A_glob[r*K+c];  // same A for both tiles in this demo
    }
    for (int i = tid; i < K*N; i += blockDim.x) {
        int r = i/N, c = i%N;
        reinterpret_cast<__half*>(smem_Ba)[(c/8)*128+(r/8)*64+(c%8)*8+r%8] = B_glob[r*N+c];
        reinterpret_cast<__half*>(smem_Bb)[(c/8)*128+(r/8)*64+(c%8)*8+r%8] = B_glob[r*N+c];  // same B for both tiles in this demo
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // ── Build descriptors ─────────────────────────────────────────
    // SBO for K-major FP16: 8 rows × K cols × 2 bytes = 8×16×2=256 for K=16
    uint64_t desc_Aa = make_desc(&smem_Aa[0][0], 8*K*2);
    uint64_t desc_Ba = make_desc(&smem_Ba[0][0], 8*K*2);
    uint64_t desc_Ab = make_desc(&smem_Ab[0][0], 8*K*2);
    uint64_t desc_Bb = make_desc(&smem_Bb[0][0], 8*K*2);

    // ── Declare FP16 accumulator registers ───────────────────────
    // D_REGS_F16=16 .b32 registers per thread, each holds 2 FP16 values.
    uint32_t dA[D_REGS_F16], dB[D_REGS_F16];
    #pragma unroll
    for (int i = 0; i < D_REGS_F16; i++) { dA[i] = 0; dB[i] = 0; }

    // ── STEP 1: Fence before first wgmma sequence ─────────────────
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    // ── STEP 2: Issue wgmma for TILE A ───────────────────────────
    // m64n64k16, f16 acc, f16 inputs. scale_d=0 (overwrite).
    // Note: f16 accumulator → 16 .b32 output registers ("+r" constraint)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
        " %16, %17, 0, 1, 1, 0, 0;\n"
        : "+r"(dA[0]),  "+r"(dA[1]),  "+r"(dA[2]),  "+r"(dA[3]),
          "+r"(dA[4]),  "+r"(dA[5]),  "+r"(dA[6]),  "+r"(dA[7]),
          "+r"(dA[8]),  "+r"(dA[9]),  "+r"(dA[10]), "+r"(dA[11]),
          "+r"(dA[12]), "+r"(dA[13]), "+r"(dA[14]), "+r"(dA[15])
        : "l"(desc_Aa), "l"(desc_Ba)
        : "memory"
    );

    // ── STEP 3: Commit tile A as Group 0 ─────────────────────────
    asm volatile("wgmma.commit_group.sync.aligned;\n" : : : "memory");

    // ── STEP 4: Prepare tile B (while Group 0 runs in background) ─
    // In a real kernel: TMA loads next tile, updates pointers, etc.
    // Here: the data is already loaded; we just demonstrate the pattern.
    // fence again before the next wgmma sequence:
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    // ── STEP 5: Issue wgmma for TILE B ───────────────────────────
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
        " %16, %17, 0, 1, 1, 0, 0;\n"
        : "+r"(dB[0]),  "+r"(dB[1]),  "+r"(dB[2]),  "+r"(dB[3]),
          "+r"(dB[4]),  "+r"(dB[5]),  "+r"(dB[6]),  "+r"(dB[7]),
          "+r"(dB[8]),  "+r"(dB[9]),  "+r"(dB[10]), "+r"(dB[11]),
          "+r"(dB[12]), "+r"(dB[13]), "+r"(dB[14]), "+r"(dB[15])
        : "l"(desc_Ab), "l"(desc_Bb)
        : "memory"
    );

    // ── STEP 6: Commit tile B as Group 1 ─────────────────────────
    asm volatile("wgmma.commit_group.sync.aligned;\n" : : : "memory");

    // ── STEP 7: wait_group<1> — tile A done, tile B may still run ─
    // After this: Group 0 (tile A) is guaranteed complete.
    //             Group 1 (tile B) may still be running.
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(1) : "memory");

    // ── STEP 8: Fence + stmatrix for tile A ──────────────────────
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    int warp = tid / 32, lane = tid % 32;
    // Each x4 call writes four 8×8 matrices: two row halves × two column halves.
    for (int slice = 0; slice < 4; ++slice) {
        int row = warp * 16 + lane % 16;
        int col = slice * 16 + (lane / 16) * 8;
        uint32_t address = (uint32_t)__cvta_generic_to_shared(&smem_out[row][col]);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
            :: "r"(address), "r"(dA[slice*4]), "r"(dA[slice*4+1]),
               "r"(dA[slice*4+2]), "r"(dA[slice*4+3]) : "memory");
    }
    __syncthreads();
    // Copy first few elements to output for verification:
    for (int i = tid; i < M * N; i += blockDim.x) {
        out_A[i] = smem_out[i/N][i%N];
    }
    __syncthreads();

    // ── STEP 9: wait_group<0> — drain tile B ─────────────────────
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(0) : "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    // Each x4 call writes four 8×8 matrices: two row halves × two column halves.
    for (int slice = 0; slice < 4; ++slice) {
        int row = warp * 16 + lane % 16;
        int col = slice * 16 + (lane / 16) * 8;
        uint32_t address = (uint32_t)__cvta_generic_to_shared(&smem_out[row][col]);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
            :: "r"(address), "r"(dB[slice*4]), "r"(dB[slice*4+1]),
               "r"(dB[slice*4+2]), "r"(dB[slice*4+3]) : "memory");
    }
    __syncthreads();
    for (int i = tid; i < M * N; i += blockDim.x) {
        out_B[i] = smem_out[i/N][i%N];
    }

    // Sanity check: all output values should be non-zero
    float sumA = 0.0f, sumB = 0.0f;
    for (int i = 0; i < D_REGS_F16; i++) {
        uint16_t lo = dA[i] & 0xFFFF;
        uint16_t hi = dA[i] >> 16;
        sumA += __half2float(*reinterpret_cast<__half*>(&lo));
        sumA += __half2float(*reinterpret_cast<__half*>(&hi));
        lo = dB[i] & 0xFFFF; hi = dB[i] >> 16;
        sumB += __half2float(*reinterpret_cast<__half*>(&lo));
        sumB += __half2float(*reinterpret_cast<__half*>(&hi));
    }
    if (sumA > 0.0f && sumB > 0.0f) atomicAdd(pass_count, 1u);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex01: wgmma commit/wait pipeline + stmatrix epilogue    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major != 9) { fprintf(stderr,"Requires Hopper SM 9.0; Blackwell needs tcgen05\n"); return 1; }

    const int A_N = M*K, B_N = K*N, D_N = M*N;
    __half *hA = new __half[A_N];
    __half *hB = new __half[B_N];
    // Audit-only exactly representable, unique-output inputs.
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            hA[r*K+k] = __float2half(k == r % 16 ? (float)(1 << (r/16)) : 0.0f);
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < N; ++c)
            hB[k*N+c] = __float2half(1.0f + (64*k+c)/1024.0f);

    __half *dA, *dB; __half *dOA, *dOB;
    uint32_t *dpass, hpass = 0;
    CHECK_CUDA(cudaMalloc(&dA,   A_N*2)); CHECK_CUDA(cudaMalloc(&dB,   B_N*2));
    CHECK_CUDA(cudaMalloc(&dOA,  D_N*2)); CHECK_CUDA(cudaMalloc(&dOB,  D_N*2));
    CHECK_CUDA(cudaMalloc(&dpass, 4));
    CHECK_CUDA(cudaMemcpy(dA, hA, A_N*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_N*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dpass, 0, 4));

    printf("Shape: m64n64k16, FP16 accumulator, 2 tiles pipelined\n");
    printf("AUDIT: 4096 unique exact FP16 outputs per tile; exact coordinate checks.\n\n");

    wgmma_pipeline_kernel<<<1, 128>>>(dA, dB, dOA, dOB, dpass);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&hpass, dpass, 4, cudaMemcpyDeviceToHost));

    __half *hOA = new __half[D_N];
    CHECK_CUDA(cudaMemcpy(hOA, dOA, D_N*2, cudaMemcpyDeviceToHost));
    printf("Threads with non-zero sums in BOTH tiles: %u / 128\n", hpass);
    printf("Sample out_A[0] = %.6f\n", __half2float(hOA[0]));

    __half *hOB = new __half[D_N];
    CHECK_CUDA(cudaMemcpy(hOB, dOB, D_N*2, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < D_N; ++i) {
        int r = i / N, c = i % N;
        float expected = (float)(1 << (r/16)) *
                         (1.0f + (64*(r%16)+c)/1024.0f);
        float got_a = __half2float(hOA[i]), got_b = __half2float(hOB[i]);
        if (got_a != expected) {
            if (errors < 8) printf("A[%d,%d]: got %.9g expected %.9g\n", r,c,got_a,expected);
            ++errors;
        }
        if (got_b != expected) {
            if (errors < 8) printf("B[%d,%d]: got %.9g expected %.9g\n", r,c,got_b,expected);
            ++errors;
        }
    }
    printf("Output verification: %d errors across both 64×64 tiles\n", errors);
    delete[] hOB;
    bool ok = hpass == 128 && errors == 0;
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", ok
           ? "✓ ALL PASS — commit/wait pipeline + stmatrix working!"
           : "✗ FAILURES — check fence/commit/wait placement");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    delete[] hA; delete[] hB; delete[] hOA;
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dOA)); CHECK_CUDA(cudaFree(dOB)); CHECK_CUDA(cudaFree(dpass));
    return ok ? 0 : 1;
}
