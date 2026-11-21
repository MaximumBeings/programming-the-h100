// ═══════════════════════════════════════════════════════════════════
// ex01_wgmma_basic.cu
//
// PURPOSE: Demonstrate wgmma.mma_async with BF16 inputs and FP32 accumulators.
//   Shows the complete pipeline:
//     1. Load A and B tiles into SMEM (simulated with host fill)
//     2. Build WGMMA descriptors for A and B
//     3. wgmma.fence before first wgmma call
//     4. wgmma.mma_async to compute the m64n64k16 tile product
//     5. wgmma.commit_group + wgmma.wait_group<0> to synchronize
//     6. Read FP32 accumulators from registers
//     7. Verify against CPU reference
//
// NOTE: This is a pedagogical example. Production GEMM kernels use
//   TMA (cp.async.bulk) to load SMEM, double-buffering with mbarrier,
//   and ldmatrix for A tile loading when A is reused.
//
// REQUIRES: Hopper sm_90a (H100/H200); not a Blackwell kernel
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} } while(0)

// Tile dimensions for this example
static constexpr int M = 64;    // fixed for all wgmma
static constexpr int N = 64;    // output columns
static constexpr int K = 16;    // inner product dimension for BF16

// Output registers per thread for m64n64k16, f32 accumulator:
// (M × N) / 128 = (64 × 64) / 128 = 32 FP32 registers per thread
static constexpr int D_REGS = (M * N) / 128;   // = 32

// ─────────────────────────────────────────────────────────────────
// Build the 64-bit WGMMA descriptor from SMEM pointer and layout info.
//
// K-major BF16, no swizzle: 8×8 core matrices, LBO=128B, SBO=256B.
// Production code (CUTLASS) uses a more general builder.
// ─────────────────────────────────────────────────────────────────
__device__ uint64_t build_wgmma_desc(const __nv_bfloat16* smem_ptr,
                                      uint32_t sbo_bytes) {
    // Get 32-bit SMEM offset (strips UVA tag bits)
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);

    // Field 1: Base address (bits 13:0) = smem_addr >> 4
    uint64_t base_field = (smem_addr >> 4) & 0x3FFF;

    // Field 2: LBO (bits 29:16) = 128 bytes / 16
    uint64_t lbo_field  = 8ULL; // 128 bytes / 16

    // Field 3: SBO (bits 45:32) = sbo_bytes / 16
    uint64_t sbo_field  = ((uint64_t)(sbo_bytes / 16)) & 0x3FFF;

    // Field 4: Matrix base offset (bits 51:49), unused without swizzle
    uint64_t base_off   = 0ULL;

    // Field 5: Swizzle mode (bits 63:62) = 0 for no swizzle
    uint64_t swizzle    = 0ULL; // match unswizzled core-matrix storage

    return base_field
         | (lbo_field << 16)
         | (sbo_field << 32)
         | (base_off  << 49)
         | (swizzle   << 62);
}

// ─────────────────────────────────────────────────────────────────
// KERNEL: one warp group performs m64n64k16 BF16→FP32 wgmma
// ─────────────────────────────────────────────────────────────────
__global__ void wgmma_kernel(
    const __nv_bfloat16* __restrict__ A_global,  // M×K BF16 matrix
    const __nv_bfloat16* __restrict__ B_global,  // K×N BF16 matrix
    float*               __restrict__ D_global,  // M×N FP32 output
    uint32_t* __restrict__ pass_count)
{
    // This kernel requires exactly 128 threads (1 warp group of 4 warps)
    // Host launch below supplies the required 128 threads.

    // ── Shared memory ─────────────────────────────────────────────
    // A tile: M×K = 64×16 BF16 = 2048 bytes → 128B swizzle needs 1024B alignment
    // B tile: K×N = 16×64 BF16 = 2048 bytes; this example is unswizzled.
    __shared__ __align__(1024) __nv_bfloat16 smem_A[M][K];
    __shared__ __align__(1024) __nv_bfloat16 smem_B[K][N];

    int tid = threadIdx.x;

    // ── Load A and B into SMEM from global (simple linear load) ───
    // In production: use TMA cp.async.bulk.tensor + mbarrier.
    // Here: simple thread-driven load for pedagogical clarity.
    for (int i = tid; i < M * K; i += blockDim.x) {
        int r = i / K, c = i % K;
        reinterpret_cast<__nv_bfloat16*>(smem_A)[(r/8)*128 + (c/8)*64 + (r%8)*8 + c%8] = A_global[r * K + c];
    }
    for (int i = tid; i < K * N; i += blockDim.x) {
        int r = i / N, c = i % N;
        reinterpret_cast<__nv_bfloat16*>(smem_B)[(c/8)*128 + (r/8)*64 + (c%8)*8 + r%8] = B_global[r * N + c];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads(); // publish generic shared stores to the async proxy

    // Both operands are staged as K-major 8×8 core matrices without swizzle.
    // The second K vector starts 128B later; the next MN core starts 256B later.
    uint64_t desc_a = build_wgmma_desc(&smem_A[0][0], 256);
    uint64_t desc_b = build_wgmma_desc(&smem_B[0][0], 256);

    // ── Declare accumulator registers ─────────────────────────────
    // 32 FP32 registers per thread for m64n64k16.
    // Initialized to zero (scale_d=0 will overwrite on first call).
    float d[D_REGS];
    #pragma unroll
    for (int i = 0; i < D_REGS; i++) d[i] = 0.0f;

    // ── STEP 1: wgmma.fence — make prior register state visible ───
    // Required before the first wgmma.mma_async in a warp group.
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    // ── STEP 2: Issue wgmma.mma_async ─────────────────────────────
    // m64n64k16: M=64, N=64, K=16, FP32 acc, BF16 inputs.
    // scale_d=0: OVERWRITE (initialize accumulator, ignore prior d values).
    // imm-scale-a=1, imm-scale-b=1: no negation.
    // imm-trans-a=0, imm-trans-b=0: no transpose.
    //
    // Output operands: 32 FP32 registers d[0]..d[31] with "+f" constraint
    // (read-write: wgmma reads the old value if scale_d=1, writes new result).
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        // Output: 32 FP32 accumulator registers
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        // A descriptor (or A registers for register variant)
        " %32,"
        // B descriptor
        " %33,"
        // scale_d=0 (overwrite), imm-scale-a=1, imm-scale-b=1, trans-a=0, trans-b=0
        " 0, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b)
        : "memory"
    );

    // ── STEP 3: Commit and wait for wgmma to complete ─────────────
    asm volatile("wgmma.commit_group.sync.aligned;\n" : : : "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(0) : "memory");

    // ── STEP 4: All groups are complete after wait_group<0>. ───────
    // This extra fence is conservative; wait_group made the outputs available.
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");

    // ── STEP 5: Write results to global memory ─────────────────────
    // The D registers are distributed across 128 threads.
    // The hardware-defined layout for m64n64k16:
    // Thread (warp w, lane l) owns specific (row, col) pairs.
    // For simplicity, we write all D registers to a flat global buffer
    // in thread order so the host can verify the computation.
    int warp   = tid >> 5;        // warp ID within warp group (0..3)
    int lane   = tid & 31;        // lane within warp (0..31)
    // Each thread writes D_REGS=32 FP32 values
    int base_idx = tid * D_REGS;
    for (int i = 0; i < D_REGS; i++) {
        D_global[base_idx + i] = d[i];
    }
    (void)warp; (void)lane;

    bool finite = true;
    for (int i = 0; i < D_REGS; i++) finite = finite && isfinite(d[i]);
    if (finite) atomicAdd(pass_count, 1u);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex01: wgmma.mma_async — m64n64k16 BF16→FP32            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major != 9) { fprintf(stderr, "Requires Hopper SM 9.0; Blackwell needs tcgen05\n"); return 1; }

    // Use nonuniform, exactly representable BF16 values to expose layout errors.
    const int A_ELEMS = M * K, B_ELEMS = K * N, D_ELEMS = 128 * D_REGS;
    __nv_bfloat16 *h_A = new __nv_bfloat16[A_ELEMS];
    __nv_bfloat16 *h_B = new __nv_bfloat16[B_ELEMS];
    for (int i = 0; i < A_ELEMS; i++) h_A[i] = __float2bfloat16((i % 17 - 8) / 16.0f);
    for (int i = 0; i < B_ELEMS; i++) h_B[i] = __float2bfloat16((i % 13 - 6) / 16.0f);
    // CPU verification uses the rounded BF16 inputs, not unrounded float literals.

    __nv_bfloat16 *d_A, *d_B;
    float *d_D; uint32_t *d_pass, h_pass = 0;
    CHECK_CUDA(cudaMalloc(&d_A, A_ELEMS * 2));
    CHECK_CUDA(cudaMalloc(&d_B, B_ELEMS * 2));
    CHECK_CUDA(cudaMalloc(&d_D, D_ELEMS * 4));
    CHECK_CUDA(cudaMalloc(&d_pass, 4));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_ELEMS*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_ELEMS*2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_pass, 0, 4));

    printf("Shape: m64n64k16, D_REGS=%d per thread, 128 threads total\n", D_REGS);
    printf("Patterned BF16 inputs; every accumulator checked against CPU GEMM.\n\n");

    // 1 block, 128 threads (exactly 1 warp group of 4 warps)
    wgmma_kernel<<<1, 128>>>(d_A, d_B, d_D, d_pass);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_pass, d_pass, 4, cudaMemcpyDeviceToHost));

    float *h_D = new float[D_ELEMS];
    CHECK_CUDA(cudaMemcpy(h_D, d_D, D_ELEMS*4, cudaMemcpyDeviceToHost));

    int elem_ok = 0;
    float max_error = 0.0f, first_expected = 0.0f;
    for (int tid = 0; tid < 128; ++tid) {
        int warp = tid / 32, lane = tid % 32;
        for (int reg = 0; reg < D_REGS; ++reg) {
            int row = warp * 16 + lane / 4 + ((reg % 4) / 2) * 8;
            int col = (reg / 4) * 8 + (lane % 4) * 2 + reg % 2;
            float expected = 0.0f;
            for (int k = 0; k < K; ++k)
                expected += __bfloat162float(h_A[row*K+k]) * __bfloat162float(h_B[k*N+col]);
            float got = h_D[tid*D_REGS+reg];
            float error = fabsf(got - expected);
            max_error = fmaxf(max_error, error);
            if (std::isfinite(got) && error < 1e-5f) ++elem_ok;
            if (tid == 0 && reg == 0) first_expected = expected;
        }
    }
    printf("Threads with finite accumulators: %u / 128\n", h_pass);
    printf("D register values correct: %d / %d\n", elem_ok, D_ELEMS);
    printf("Max absolute error: %.8f\n", max_error);
    printf("Sample d[0] = %.6f (expected %.6f)\n", h_D[0], first_expected);

    bool ok = (h_pass == 128 && elem_ok == D_ELEMS);
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", ok
           ? "✓ ALL PASS — wgmma computes correct BF16→FP32 result!"
           : "✗ FAILURES — check descriptor or register layout");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    delete[] h_A; delete[] h_B; delete[] h_D;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D)); CHECK_CUDA(cudaFree(d_pass));
    return ok ? 0 : 1;
}
