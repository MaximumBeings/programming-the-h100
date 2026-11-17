// ═══════════════════════════════════════════════════════════════════
// ex01_cluster_inspect.cu
//
// PURPOSE: Demonstrates Thread Block Clusters on the H100.
//   - Shows how to launch a cluster kernel
//   - Reads cluster special registers via inline PTX
//   - Demonstrates cooperative_groups cluster API
//   - Performs a simple cross-block shared memory exchange
//     to prove DSMEM access works
//
// WHAT TO LOOK FOR IN THE OUTPUT:
//   - Each block prints its global ID (blockIdx.x) and its rank
//     within the cluster (0 or 1 for a size-2 cluster)
//   - The cross-block exchange shows each block reading the other's
//     shared memory value
//
// REQUIRES: sm_90a (H100/Hopper) or sm_100a (B200/Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cooperative_groups.h>    // for cg::this_cluster(), cluster_group

namespace cg = cooperative_groups;

// ─── Error-checking macro ──────────────────────────────────────────
// Always wrap CUDA API calls in this. It prints file, line, and the
// error string so you know exactly where something went wrong.
#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// ─── Result struct ─────────────────────────────────────────────────
// We collect per-block info on the device and copy it back to the host
// for printing. Using a struct keeps the data organized.
struct BlockInfo {
    uint32_t global_block_id;   // blockIdx.x
    uint32_t cluster_rank;      // this block's rank within its cluster
    uint32_t cluster_size;      // total blocks in the cluster
    uint32_t cluster_ctaid_x;   // cluster-local block ID (same as rank for 1D)
    float    my_smem_value;     // value this block wrote to its SMEM
    float    partner_smem_value; // value this block read from partner's SMEM
};

// ─── Cluster kernel ────────────────────────────────────────────────
// __cluster_dims__(2, 1, 1) tells the compiler to group blocks into
// clusters of 2×1×1 = 2 blocks. This attribute is baked into the PTX.
//
// The kernel does:
//   1. Read cluster special registers via inline PTX
//   2. Write a unique value to this block's SMEM
//   3. Sync across the cluster (ensures partner's SMEM is ready)
//   4. Read the partner block's SMEM via DSMEM
//   5. Write all info to global output array
__global__ void __cluster_dims__(2, 1, 1)
inspect_cluster(BlockInfo* output) {

    // ── Step 1: Read cluster registers ──────────────────────────
    // These inline PTX statements read hardware registers directly.
    // They are equivalent to calling cg::this_cluster() methods.
    uint32_t global_id, rank, cluster_size, ctaid_x;

    // blockIdx.x — global block index across the entire grid
    asm("mov.u32 %0, %ctaid.x;"          : "=r"(global_id));
    // Rank within the cluster (0 to cluster_size-1)
    asm("mov.u32 %0, %cluster_ctarank;"  : "=r"(rank));
    // Total blocks in this cluster
    asm("mov.u32 %0, %cluster_nctarank;" : "=r"(cluster_size));
    // This block's position within cluster (same as rank for 1D)
    asm("mov.u32 %0, %cluster_ctaid.x;"  : "=r"(ctaid_x));

    // ── Step 2: Shared memory setup ─────────────────────────────
    // Each block allocates 64 floats of SMEM.
    // All 64 floats are initialized to the same value so we can
    // check any index after the exchange.
    __shared__ float smem[64];

    // Write a unique value: global block ID × 100
    // Block 0 writes 0.0, Block 1 writes 100.0, Block 2 writes 200.0, ...
    // This lets us verify which block's data we're reading.
    float my_val = static_cast<float>(global_id * 100);
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        smem[i] = my_val;
    }
    // __syncthreads ensures ALL threads in THIS block have finished
    // writing their SMEM before we proceed. Without this, the cluster
    // sync below might fire before some threads have written.
    __syncthreads();

    // ── Step 3: Cluster-wide synchronization ────────────────────
    // cluster.sync() is like __syncthreads() but for ALL blocks in
    // the cluster. After this point, EVERY block's SMEM writes are
    // visible to every other block in the cluster.
    //
    //  Timeline:
    //  Block 0: ... writes smem ... | cluster.sync() | reads Block 1's smem
    //  Block 1: ... writes smem ... | cluster.sync() | reads Block 0's smem
    //                                        ↑
    //                           Both blocks MUST reach here before either
    //                           can proceed. This is the safety guarantee.
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();

    // ── Step 4: Read partner's SMEM via DSMEM ───────────────────
    // map_shared_rank(ptr, target_rank) translates 'ptr' (which points
    // into THIS block's SMEM) into an equivalent pointer that accesses
    // the SAME OFFSET in the TARGET block's SMEM.
    //
    //  Example for block with rank 0:
    //    partner_rank = 1 - 0 = 1
    //    partner_ptr points to Block 1's smem[0]
    //    partner_ptr[5] accesses Block 1's smem[5]
    //
    // Internally, this uses the 'mapa' PTX instruction (Section 16).
    uint32_t partner_rank = 1 - rank;  // 0→1, 1→0
    float* partner_smem = cluster.map_shared_rank(smem, partner_rank);
    float partner_val = partner_smem[0];  // read index 0 from partner's SMEM

    // ── Step 5: Write results to global memory ──────────────────
    // Only thread 0 in each block writes the output to avoid races.
    if (threadIdx.x == 0) {
        output[global_id].global_block_id    = global_id;
        output[global_id].cluster_rank       = rank;
        output[global_id].cluster_size       = cluster_size;
        output[global_id].cluster_ctaid_x    = ctaid_x;
        output[global_id].my_smem_value      = my_val;
        output[global_id].partner_smem_value = partner_val;
    }
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Thread Block Cluster Inspection — H100 Demo         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Check that we have a Hopper GPU (compute capability 9.0)
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name,
           prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr,
            "ERROR: This example requires a Hopper GPU (SM 9.0+).\n"
            "       Your GPU is SM %d.%d.\n"
            "       Use an H100 or B200 instance.\n",
            prop.major, prop.minor);
        return 1;
    }

    // ── Launch configuration ─────────────────────────────────────
    // 4 blocks total, cluster size 2 → 2 clusters of 2 blocks each
    //   Cluster 0: blocks 0 and 1
    //   Cluster 1: blocks 2 and 3
    //
    // Grid MUST be a multiple of cluster size (4 is divisible by 2 ✓)
    const int NUM_BLOCKS = 4;
    dim3 grid(NUM_BLOCKS, 1, 1);
    dim3 block(32, 1, 1);   // 32 threads per block (small for demo)

    // ── Allocate output buffer ───────────────────────────────────
    BlockInfo* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, NUM_BLOCKS * sizeof(BlockInfo)));

    // ── Launch cluster kernel ────────────────────────────────────
    inspect_cluster<<<grid, block>>>(d_output);
    CHECK_CUDA(cudaGetLastError());       // check launch errors
    CHECK_CUDA(cudaDeviceSynchronize());  // wait for kernel to finish

    // ── Copy results back to host ────────────────────────────────
    BlockInfo h_output[NUM_BLOCKS];
    CHECK_CUDA(cudaMemcpy(h_output, d_output,
                          NUM_BLOCKS * sizeof(BlockInfo),
                          cudaMemcpyDeviceToHost));

    // ── Print per-block report ───────────────────────────────────
    printf("Per-Block Report:\n");
    printf("%-12s %-14s %-14s %-12s %-12s %-16s\n",
           "Block(global)", "Cluster Rank", "Cluster Size",
           "My SMEM Val", "Partner Val", "Exchange OK?");
    printf("%s\n", std::string(80, '-').c_str());

    int tests = 0, passed = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        BlockInfo& b = h_output[i];

        // Expected partner value:
        //   Block 0 (rank 0) → reads Block 1 → value = 1 * 100 = 100.0
        //   Block 1 (rank 1) → reads Block 0 → value = 0 * 100 = 0.0
        //   Block 2 (rank 0) → reads Block 3 → value = 3 * 100 = 300.0
        //   Block 3 (rank 1) → reads Block 2 → value = 2 * 100 = 200.0
        int partner_global_id = (b.cluster_rank == 0)
            ? (b.global_block_id + 1)   // rank 0 reads from rank 1
            : (b.global_block_id - 1);  // rank 1 reads from rank 0
        float expected_partner = static_cast<float>(partner_global_id * 100);
        bool ok = (b.partner_smem_value == expected_partner);

        printf("  %-12u %-14u %-14u %-12.0f %-12.0f %s\n",
               b.global_block_id, b.cluster_rank, b.cluster_size,
               b.my_smem_value, b.partner_smem_value,
               ok ? "✓ PASS" : "✗ FAIL");
        tests++;
        if (ok) passed++;
    }

    printf("\n");
    printf("DSMEM Exchange Tests: %d/%d passed\n", passed, tests);

    // ── Verify cluster metadata ──────────────────────────────────
    printf("\nCluster Metadata Verification:\n");
    int meta_ok = 1;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        if (h_output[i].cluster_size != 2) {
            printf("  FAIL: block %d cluster_size = %u (expected 2)\n",
                   i, h_output[i].cluster_size);
            meta_ok = 0;
        }
        // Blocks 0,1 are cluster 0; blocks 2,3 are cluster 1
        // Ranks within cluster: block 0 → rank 0, block 1 → rank 1, etc.
        uint32_t expected_rank = h_output[i].global_block_id % 2;
        if (h_output[i].cluster_rank != expected_rank) {
            printf("  FAIL: block %d cluster_rank = %u (expected %u)\n",
                   i, h_output[i].cluster_rank, expected_rank);
            meta_ok = 0;
        }
    }
    if (meta_ok) printf("  ✓ All cluster metadata correct\n");

    CHECK_CUDA(cudaFree(d_output));

    printf("\n════════════════════════════════════════════════\n");
    if (passed == tests && meta_ok) {
        printf("ALL TESTS PASSED — Clusters working correctly!\n");
    } else {
        printf("SOME TESTS FAILED — Check error messages above.\n");
    }
    printf("════════════════════════════════════════════════\n");

    return (passed == tests && meta_ok) ? 0 : 1;
}
