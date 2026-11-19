// ═══════════════════════════════════════════════════════════════════
// ex03_state_spaces.cu
//
// PURPOSE: Demonstrate all major CUDA/PTX state spaces with explicit
//   PTX loads and stores for each. Each test shows one state space.
//
// LAYOUT:
//   Test 1: .reg  — Register values (trivial, shows constraint types)
//   Test 2: .shared — Shared memory load/store via PTX
//   Test 3: .global — Global memory load/store via explicit PTX addr
//   Test 4: .const — Constant memory broadcast
//   Test 5: .local — Forced register spill to local memory (diagnostic)
//   Test 6: Combining all spaces in one kernel (realistic pattern)
//
// WORKS ON: sm_80+ (all examples, no Hopper-specific features)
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

// ─── Constant memory declaration ─────────────────────────────────
// Must be at FILE SCOPE. The compiler places this in .const space.
// All threads see the same values, and if they all read the same
// address simultaneously, it's served in 1 cycle from const cache.
__constant__ float SCALE_FACTOR[4] = {2.0f, 3.0f, 4.0f, 5.0f};

// ═══════════════════════════════════════════════════════════════════
// TEST 2: Shared Memory — explicit PTX load and store
// ═══════════════════════════════════════════════════════════════════
//
// Shows:
//   (a) How to get a shared memory address using __cvta_generic_to_shared
//   (b) How to write to SMEM via asm volatile st.shared
//   (c) How to read from SMEM via asm volatile ld.shared
//   (d) How __syncthreads() maps to bar.sync 0 in PTX
//
// The "0" in "bar.sync 0" is the barrier ID.
// H100 supports barriers 0-15. Using 0 is the default.
__global__ void test_shared_space(float* output) {
    // Each thread has its own shared memory slot
    __shared__ float smem[256];

    // ── Address calculation ─────────────────────────────────────
    // __cvta_generic_to_shared strips the high bits from the C++ pointer,
    // producing a 32-bit offset into shared memory.
    //
    // Why 32-bit? Because shared memory is only 228KB — fits easily
    // in 32 bits (max value ~233,472 < 4,294,967,296).
    // Global pointers are 64-bit because HBM is 80GB.
    uint32_t smem_base = __cvta_generic_to_shared(smem);
    // smem_base is the byte offset of smem[0] within shared memory.
    // smem_base + threadIdx.x * 4 is the byte offset of smem[threadIdx.x].

    uint32_t my_slot_addr = smem_base + threadIdx.x * sizeof(float);

    // ── Write to SMEM via PTX store ─────────────────────────────
    // Value = thread index × 2.0 + 0.5 (unique per thread)
    float my_val = threadIdx.x * 2.0f + 0.5f;
    asm volatile(
        "st.shared.f32 [%0], %1;"  // store float to shared memory
        :                           // no outputs (store has side effects only)
        : "r"(my_slot_addr),        // %0 = address (32-bit for SMEM!)
          "f"(my_val)               // %1 = value to store
        : "memory"                  // memory clobber: tells compiler memory changed
    );

    // ── Barrier — wait for all threads to finish storing ────────
    // This maps to: bar.sync 0; in PTX
    // Every thread must reach this point before any thread proceeds.
    __syncthreads();

    // ── Read from SMEM via PTX load ─────────────────────────────
    // Each thread reads from the NEXT thread's slot (circular).
    // Thread 0 reads smem[1], thread 1 reads smem[2], ..., last reads smem[0].
    uint32_t neighbor = (threadIdx.x + 1) % blockDim.x;
    uint32_t neighbor_addr = smem_base + neighbor * sizeof(float);

    float neighbor_val;
    asm volatile(
        "ld.shared.f32 %0, [%1];"  // load float from shared memory
        : "=f"(neighbor_val)        // %0 = destination (32-bit float register)
        : "r"(neighbor_addr)        // %1 = address (32-bit)
    );

    // Thread 0 writes its result to global memory
    if (threadIdx.x == 0) {
        // Thread 0 read from slot 1, which holds: 1 * 2.0 + 0.5 = 2.5
        output[0] = neighbor_val;
    }
}

// ═══════════════════════════════════════════════════════════════════
// TEST 3: Global Memory — explicit 64-bit address PTX
// ═══════════════════════════════════════════════════════════════════
//
// Shows how global memory uses 64-bit addresses ("l" constraint)
// and demonstrates vectorized loads (v4.f32 = 4 floats at once).
//
// Vectorized loads improve bandwidth utilization because:
//   - One 128-bit transaction vs four 32-bit transactions
//   - The memory system is more efficient with wider requests
//   - Coalescing is achieved more easily with wider accesses
__global__ void test_global_space(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ── Standard scalar load via PTX ────────────────────────────
    // Compute 64-bit byte address of input[idx]
    uint64_t in_addr = reinterpret_cast<uint64_t>(input) + idx * sizeof(float);
    // "l" constraint = 64-bit register (use for global pointers!)
    float val;
    asm volatile("ld.global.f32 %0, [%1];"
                 : "=f"(val)
                 : "l"(in_addr));

    // ── Scale by 10 ─────────────────────────────────────────────
    float result;
    asm("mul.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(val), "f"(10.0f));

    // ── Store result via PTX ────────────────────────────────────
    uint64_t out_addr = reinterpret_cast<uint64_t>(output) + idx * sizeof(float);
    asm volatile("st.global.f32 [%0], %1;"
                 :
                 : "l"(out_addr), "f"(result)
                 : "memory");
}

// ═══════════════════════════════════════════════════════════════════
// TEST 4: Constant Memory — broadcast read
// ═══════════════════════════════════════════════════════════════════
//
// All threads read from __constant__ SCALE_FACTOR[].
// When all threads in a warp read the SAME index, it's 1 cycle.
// When they read different indices, it serializes (~32 cycles).
//
// This test has all threads read SCALE_FACTOR[0] = 2.0f (broadcast).
__global__ void test_const_space(float* output) {
    // In CUDA C++, reading __constant__ data looks like a normal array access.
    // Under the hood, the compiler emits ld.const.f32.
    // We can also read it explicitly via PTX:
    float scale;
    asm("ld.const.f32 %0, [SCALE_FACTOR];"  // load SCALE_FACTOR[0]
        : "=f"(scale)
    );
    // Note: PTX constant memory uses the SYMBOL NAME directly.
    // SCALE_FACTOR[1] would be: [SCALE_FACTOR + 4]  (byte offset)

    // Alternatively (using C++ directly, same PTX is generated):
    float scale_cpp = SCALE_FACTOR[0];

    if (threadIdx.x == 0) {
        output[0] = scale;       // from PTX ld.const
        output[1] = scale_cpp;   // from C++ access (same result)
    }
}

// ═══════════════════════════════════════════════════════════════════
// TEST 6: Combined Kernel — Realistic Usage of All State Spaces
// ═══════════════════════════════════════════════════════════════════
//
// This is what a real GPU kernel looks like. It uses:
//   .global → input data loading (expensive, done once per block)
//   .shared → tile storage (cheap, accessed many times)
//   .const  → read-only parameters (broadcast, cheap)
//   .reg    → intermediate computation (free)
//   .global → output writing
//
// Pattern: Load tile from global → store in SMEM → compute using
//          constant scale → write results back to global.
//
// This is the fundamental template for virtually every high-performance
// GPU kernel: global→SMEM tiling reduces global memory accesses.
__global__ void test_combined(float* input, float* output, int N) {
    // ── SMEM allocation ─────────────────────────────────────────
    // Each block allocates a tile of 128 floats in shared memory.
    // All threads in the block cooperatively load this tile.
    __shared__ float tile[128];

    int block_start = blockIdx.x * 128;  // first element this block handles
    int tid = threadIdx.x;

    // ── Collaborative global→SMEM load ──────────────────────────
    // Each thread loads ONE element from global into SMEM.
    // Together, all blockDim.x threads fill the entire tile.
    if (block_start + tid < N) {
        uint64_t g_addr = reinterpret_cast<uint64_t>(input + block_start + tid);
        float val;
        asm volatile("ld.global.f32 %0, [%1];"
                     : "=f"(val) : "l"(g_addr));

        uint32_t smem_addr = __cvta_generic_to_shared(tile) + tid * sizeof(float);
        asm volatile("st.shared.f32 [%0], %1;"
                     : : "r"(smem_addr), "f"(val) : "memory");
    }

    // ── Synchronize ─────────────────────────────────────────────
    // All threads must finish storing before any thread reads SMEM.
    __syncthreads();

    // ── Compute using SMEM + constant memory ────────────────────
    // Now each thread reads from SMEM (cheap) and applies a scale
    // from constant memory (also cheap, broadcast).
    if (block_start + tid < N) {
        // Read from SMEM (cheap, ~10 cycles)
        uint32_t smem_addr = __cvta_generic_to_shared(tile) + tid * sizeof(float);
        float smem_val;
        asm volatile("ld.shared.f32 %0, [%1];"
                     : "=f"(smem_val) : "r"(smem_addr));

        // Read scale from constant memory (broadcast, ~1 cycle)
        float scale = SCALE_FACTOR[0];  // C++ reads __constant__ as ld.const

        // Compute: result = smem_val * scale
        float result;
        asm("mul.f32 %0, %1, %2;" : "=f"(result) : "f"(smem_val), "f"(scale));

        // Write to global memory (expensive, but only once per element)
        uint64_t out_addr = reinterpret_cast<uint64_t>(output + block_start + tid);
        asm volatile("st.global.f32 [%0], %1;"
                     : : "l"(out_addr), "f"(result) : "memory");
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  PTX State Spaces — Test Suite                        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int tests = 0, passed = 0;

    // ─── Test 2: Shared memory ─────────────────────────────────
    printf("Test 2: .shared — Thread 0 reads thread 1's slot (expected 2.5)\n");
    {
        float *d_out, h_out;
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
        test_shared_space<<<1, 64>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        // Thread 1's value = 1 * 2.0 + 0.5 = 2.5
        tests++;
        if (fabsf(h_out - 2.5f) < 1e-5f) {
            passed++; printf("  ✓ PASS: thread 0 read %.1f from thread 1's slot\n", h_out);
        } else printf("  ✗ FAIL: got %.4f, expected 2.5\n", h_out);
        CHECK_CUDA(cudaFree(d_out));
    }

    // ─── Test 3: Global memory ─────────────────────────────────
    printf("\nTest 3: .global — Multiply by 10 via explicit PTX\n");
    {
        const int N = 8;
        float h_in[N]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float h_out[N] = {};
        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        test_global_space<<<1, N>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_out[i] - h_in[i] * 10.0f) > 1e-4f) {
                printf("  ✗ FAIL at [%d]: got %.1f, expected %.1f\n",
                       i, h_out[i], h_in[i] * 10.0f);
                ok = false;
            }
        }
        tests++;
        if (ok) { passed++; printf("  ✓ PASS: all 8 values multiplied by 10 correctly\n"); }

        CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    }

    // ─── Test 4: Constant memory ───────────────────────────────
    printf("\nTest 4: .const — Read SCALE_FACTOR[0] = 2.0 via PTX and C++\n");
    {
        float *d_out, h_out[2];
        CHECK_CUDA(cudaMalloc(&d_out, 2 * sizeof(float)));
        test_const_space<<<1, 32>>>(d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost));

        tests++;
        if (fabsf(h_out[0] - 2.0f) < 1e-5f && fabsf(h_out[1] - 2.0f) < 1e-5f) {
            passed++;
            printf("  ✓ PASS: PTX read = %.1f, C++ read = %.1f (both = 2.0)\n",
                   h_out[0], h_out[1]);
        } else {
            printf("  ✗ FAIL: PTX=%.4f, C++=%.4f, expected both 2.0\n",
                   h_out[0], h_out[1]);
        }
        CHECK_CUDA(cudaFree(d_out));
    }

    // ─── Test 6: Combined kernel ────────────────────────────────
    printf("\nTest 6: Combined — global→SMEM→compute with SCALE_FACTOR[0]=2.0\n");
    {
        const int N = 128;
        float h_in[N], h_out[N], h_expected[N];
        for (int i = 0; i < N; i++) { h_in[i] = (float)i; h_expected[i] = i * 2.0f; }

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        test_combined<<<1, N>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_out[i] - h_expected[i]) > 1e-4f) { ok = false; break; }
        }
        tests++;
        if (ok) {
            passed++;
            printf("  ✓ PASS: 128 elements scaled by 2.0 (sample: in=%.0f out=%.0f)\n",
                   h_in[5], h_out[5]);
        } else printf("  ✗ FAIL\n");

        CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  Results: %d/%d tests passed                            ║\n", passed, tests);
    printf("╚══════════════════════════════════════════════════════╝\n");
    return (passed == tests) ? 0 : 1;
}
