// ═══════════════════════════════════════════════════════════════════
// ex06_comprehensive.cu
//
// CAPSTONE EXAMPLE — combines all concepts from Lesson 2:
//   ✓ Thread Block Clusters (__cluster_dims__)
//   ✓ Cooperative Groups (cg::this_cluster())
//   ✓ Distributed Shared Memory (map_shared_rank / DSMEM)
//   ✓ Inline PTX assembly (ld.global, st.global, ld.shared, fma)
//   ✓ State space pointers (__cvta_generic_to_shared)
//   ✓ Data types (FP32 arithmetic in PTX)
//   ✓ Shared memory bank-conflict-free access pattern
//   ✓ Full test harness with host verification
//
// WHAT THE KERNEL DOES:
//   Distributed vector add: two blocks cooperate to add two halves
//   of a vector. Block 0 owns input[0..63], Block 1 owns input[64..127].
//   After a cluster-synchronised DSMEM exchange, each block writes
//   the pairwise sum to output. Result verified on host.
//
// ARCHITECTURE DIFFERENCES (Hopper vs Blackwell):
//   Hopper  (sm_90a): compile flag only, all code identical
//   Blackwell (sm_100a): compile flag only, all code identical
//   The cluster API, mapa, and DSMEM semantics are unchanged.
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA ERROR %s:%d: %s\n", \
                __FILE__,__LINE__,cudaGetErrorString(_e)); \
        exit(1); } } while(0)

// ─────────────────────────────────────────────────────────────────
// KERNEL: distributed_vector_add
//
// Launch config: cluster_size=2, grid=2 blocks, block=64 threads
//
// Each block:
//   1. Loads 64 floats from global memory into its SMEM via PTX ld.global
//   2. cluster.sync() — ensures both blocks' SMEM is ready
//   3. Reads partner block's SMEM via map_shared_rank (DSMEM)
//   4. Adds own value + partner value via PTX fma
//   5. Stores result to global output via PTX st.global
// ─────────────────────────────────────────────────────────────────
__global__ void __cluster_dims__(2, 1, 1)
distributed_vector_add(const float* __restrict__ input,
                        float*       __restrict__ output,
                        int N) {

    // ── Step 1: Cluster handle ────────────────────────────────────
    // cluster_group gives us block_rank() and map_shared_rank().
    // block_rank() = 0 for Block 0, 1 for Block 1.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int my_rank  = cluster.block_rank();      // 0 or 1
    unsigned int tid      = threadIdx.x;               // 0..63
    unsigned int gid      = blockIdx.x * blockDim.x + tid; // global thread id

    // ── Step 2: Shared memory ─────────────────────────────────────
    // Each block allocates 64 floats of SMEM.
    // Block 0's SMEM will hold input[0..63].
    // Block 1's SMEM will hold input[64..127].
    __shared__ float smem[64];

    // ── Step 3: Load from global memory via PTX ───────────────────
    // We use explicit PTX ld.global to demonstrate the "l" constraint.
    // The alternative (smem[tid] = input[gid]) generates the same PTX.
    if (gid < N) {
        // Compute 64-bit byte address of input[gid]
        uint64_t g_addr = reinterpret_cast<uint64_t>(input) +
                          (uint64_t)gid * sizeof(float);
        float val;
        asm volatile("ld.global.f32 %0, [%1];"
                     : "=f"(val)
                     : "l"(g_addr));   // "l" = 64-bit for global!

        // Store to SMEM via PTX st.shared
        // Get the 32-bit SMEM byte offset for smem[tid]
        uint32_t s_addr = __cvta_generic_to_shared(smem)
                          + tid * sizeof(float);
        asm volatile("st.shared.f32 [%0], %1;"
                     : : "r"(s_addr), "f"(val) : "memory");
    }

    // ── Step 4: Sync within the block ────────────────────────────
    // All threads in THIS block must have written their SMEM slot
    // before we do the cluster-wide sync.
    __syncthreads();

    // ── Step 5: Cluster-wide synchronization ─────────────────────
    // After cluster.sync(), BOTH blocks' SMEM writes are visible
    // to all threads in the cluster. This is the DSMEM safety guarantee.
    //
    //  Block 0 timeline:  write smem ... | cluster.sync() | read Block 1's smem
    //  Block 1 timeline:  write smem ... | cluster.sync() | read Block 0's smem
    //
    // Neither block can proceed past cluster.sync() until BOTH blocks
    // have reached it. This prevents reading stale SMEM from the partner.
    cluster.sync();

    // ── Step 6: Read partner's SMEM via DSMEM ────────────────────
    // map_shared_rank(smem, rank) translates our local smem pointer
    // into a pointer that accesses the TARGET block's smem at the
    // same element offset. Internally uses the mapa PTX instruction.
    //
    // For Block 0: partner_rank = 1 → reads Block 1's smem
    // For Block 1: partner_rank = 0 → reads Block 0's smem
    unsigned int partner_rank = 1 - my_rank;
    float* partner_smem = cluster.map_shared_rank(smem, partner_rank);

    float my_val, partner_val;

    // Read own SMEM value
    uint32_t my_s_addr = __cvta_generic_to_shared(smem) + tid * sizeof(float);
    asm volatile("ld.shared.f32 %0, [%1];"
                 : "=f"(my_val) : "r"(my_s_addr));

    // Read partner SMEM — this goes through DSMEM interconnect (~50-80 cycles)
    // partner_smem[tid] uses ld.shared::cluster internally
    partner_val = partner_smem[tid];

    // ── Step 7: Compute sum using PTX FMA ────────────────────────
    // fma.rn.f32(a, b, c) = a*b + c
    // We want: result = my_val + partner_val = my_val * 1.0 + partner_val
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(my_val), "f"(1.0f), "f"(partner_val));

    // ── Step 8: Store result to global output ─────────────────────
    if (gid < N) {
        uint64_t out_addr = reinterpret_cast<uint64_t>(output) +
                            (uint64_t)gid * sizeof(float);
        asm volatile("st.global.f32 [%0], %1;"
                     : : "l"(out_addr), "f"(result) : "memory");
    }
}

// ─────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex06: Comprehensive Capstone — Clusters + PTX + DSMEM   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ── Check hardware ────────────────────────────────────────────
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        fprintf(stderr,
            "\nERROR: Cluster features require Hopper (SM 9.0+) or Blackwell.\n"
            "       Your GPU is SM %d.%d.\n"
            "       For A100 (SM 8.0): remove cluster code and use sm_80.\n",
            prop.major, prop.minor);
        return 1;
    }
    printf("       ✓ Hopper/Blackwell GPU detected — clusters supported.\n\n");

    // ── Problem setup ─────────────────────────────────────────────
    // N = 128 elements = 2 blocks × 64 threads
    // Block 0 owns elements [0..63], Block 1 owns elements [64..127]
    const int N = 128;
    const int BLOCK_SIZE = 64;
    const int NUM_BLOCKS = N / BLOCK_SIZE;  // = 2 (one per cluster block)

    // ── Host data ─────────────────────────────────────────────────
    float h_input[N], h_output[N], h_expected[N];
    for (int i = 0; i < N; i++) h_input[i] = (float)i;

    // Expected: output[i] = input[i] + input[i ^ 64]
    // i.e. Block 0 element i pairs with Block 1 element i:
    //   output[0]  = 0  + 64  = 64
    //   output[1]  = 1  + 65  = 66
    //   output[63] = 63 + 127 = 190
    //   output[64] = 64 + 0   = 64  (Block 1 reads Block 0's value)
    //   output[65] = 65 + 1   = 66
    for (int i = 0; i < N; i++) {
        int partner_gid = (i < 64) ? (i + 64) : (i - 64);
        h_expected[i] = h_input[i] + h_input[partner_gid];
    }

    // ── Device allocation ─────────────────────────────────────────
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, N * sizeof(float)));

    // ── Launch cluster kernel ─────────────────────────────────────
    // cluster_dims(2,1,1) is set via __cluster_dims__ on the kernel.
    // Grid must be a multiple of cluster size: 2 blocks / 2 per cluster = 1 cluster.
    dim3 grid(NUM_BLOCKS, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    printf("Launch config:\n");
    printf("  Grid:    %d blocks\n", NUM_BLOCKS);
    printf("  Block:   %d threads\n", BLOCK_SIZE);
    printf("  Cluster: 2 blocks per cluster (1 cluster total)\n");
    printf("  N:       %d elements\n\n", N);

    distributed_vector_add<<<grid, block>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ── Copy results back ─────────────────────────────────────────
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ── Verify results ────────────────────────────────────────────
    printf("Verification (first 8 and last 4 elements):\n");
    printf("  %-8s %-12s %-12s %-12s %s\n",
           "Index", "Input", "Expected", "Got", "OK?");
    printf("  %s\n", std::string(60,'-').c_str());

    int errors = 0;
    // Print a sample of values
    int print_indices[] = {0,1,2,3,4,5,6,7, 63,64,65,126,127};
    for (int idx : print_indices) {
        bool ok = (fabsf(h_output[idx] - h_expected[idx]) < 1e-4f);
        if (!ok) errors++;
        printf("  %-8d %-12.1f %-12.1f %-12.1f %s\n",
               idx, h_input[idx], h_expected[idx], h_output[idx],
               ok ? "✓" : "✗ FAIL");
    }

    // Check ALL elements
    int total_errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_output[i] - h_expected[i]) > 1e-4f) total_errors++;
    }

    printf("\n");
    if (total_errors == 0) {
        printf("  ✓ ALL %d elements correct!\n", N);
        printf("\n  What just happened (review):\n");
        printf("  ├── Block 0 loaded input[0..63] into its SMEM (PTX ld.global)\n");
        printf("  ├── Block 1 loaded input[64..127] into its SMEM (PTX ld.global)\n");
        printf("  ├── cluster.sync() ensured both blocks' SMEM was ready\n");
        printf("  ├── Each block read the partner's SMEM via map_shared_rank (DSMEM)\n");
        printf("  ├── PTX fma.rn.f32 computed the pairwise sum\n");
        printf("  └── Results written back to global memory (PTX st.global)\n");
    } else {
        printf("  ✗ %d/%d elements FAILED\n", total_errors, N);
    }

    // ── Cleanup ───────────────────────────────────────────────────
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    if (total_errors == 0)
        printf("║  ✓ ALL TESTS PASSED — Comprehensive example complete!     ║\n");
    else
        printf("║  ✗ FAILURES DETECTED — Check output above                 ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return (total_errors == 0) ? 0 : 1;
}
