// ═══════════════════════════════════════════════════════════════════
// ex01_mbarrier_basic.cu
//
// PURPOSE: Demonstrate the complete mbarrier pipeline with a simulated
//   async copy. Uses cp.async (non-bulk) for portability, but shows
//   the full mbarrier init/arrive.expect_tx/wait/reuse pattern.
//
// NOTE: This uses cp.async (Ampere-style) for the actual copy but
//   wraps it in the mbarrier pattern. In production code on H100,
//   replace cp.async with cp.async.bulk.tensor + TMA descriptors.
//   The mbarrier semantics are identical.
//
// WHAT THIS DEMONSTRATES:
//   1. mbarrier initialization in shared memory
//   2. arrive.expect_tx: setting expected byte count
//   3. try_wait.parity: non-blocking completion polling
//   4. Phase tracking with local register
//   5. Barrier reuse across multiple iterations
//
// REQUIRES: sm_90a (H100) or sm_100a (Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); } } while(0)

// Number of floats per tile (must fit in SMEM)
static constexpr int TILE_SIZE  = 256;   // elements
static constexpr int TILE_BYTES = TILE_SIZE * sizeof(float);  // 1024 bytes

// ─────────────────────────────────────────────────────────────────
// KERNEL: mbarrier_pipeline_kernel
//
// Iterates over K tiles. For each tile:
//   Producer (thread 0): sets up mbarrier, issues cp.async copy
//   All threads: poll mbarrier until copy is complete
//   All threads: verify the loaded data is correct
// ─────────────────────────────────────────────────────────────────
__global__ void mbarrier_pipeline_kernel(
    const float* __restrict__ gmem_input,   // source: K tiles in global memory
    uint32_t*    __restrict__ pass_count,   // output: number of passing checks
    int K)                                  // number of tiles to process
{
    // ── Shared memory layout ─────────────────────────────────────
    // mbarrier object: MUST be 8-byte aligned (use __align__)
    // smem tile: the destination for our async copies
    __shared__ __align__(8) uint64_t mbar;
    __shared__ __align__(16) float smem[TILE_SIZE];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // ── STEP 1: Initialize mbarrier (thread 0 only) ───────────────
    // Expected thread arrival count = 1.
    // Only the producer thread (thread 0) will call arrive.expect_tx,
    // which decrements the thread count AND sets the tx byte count.
    // So the barrier completes when:
    //   (a) thread 0 has called arrive.expect_tx  [thread count 1→0]
    //   (b) the async copy delivers TILE_BYTES bytes [tx count→0]
    if (tid == 0) {
        uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            : : "r"(mbar_addr), "r"(1u)   // 1 expected thread arrival
        );
    }
    // CRITICAL: All threads must wait for init to complete before
    // anyone tries to use the barrier.
    // Publish initialized barrier state to the bulk-async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // ── Phase tracking — LOCAL REGISTER (not SMEM!) ────────────────
    // Each thread keeps its own copy of the current phase.
    // They all start at 0 and XOR to 1 after each barrier completion.
    int phase = 0;

    // ── Pipeline loop over K tiles ────────────────────────────────
    for (int k = 0; k < K; k++) {

        // STEP 2 & 3: Producer thread sets up barrier and issues copy
        if (tid == 0) {
            uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);

            // arrive.expect_tx: two effects in one instruction:
            //   1. Decrements thread arrival count: 1 → 0
            //   2. Adds TILE_BYTES to expected transaction count: 0 → 1024
            // Also has .release semantics (ordering for prior stores).
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                : : "r"(mbar_addr), "r"((uint32_t)TILE_BYTES)
            );

            // Now issue the actual async copy.
            // In production: replace with cp.async.bulk.tensor + TMA descriptor.
            // Here: use Hopper raw bulk copies linked to transaction completion.
            const float* src = gmem_input + k * TILE_SIZE;
            // The raw bulk load copies 16-byte aligned pieces into shared memory.
            // We link it to the mbarrier so hardware decrements tx count.
            // (Using a loop of cp.async to simulate bulk behavior.)
            // NOTE: In sm_90a, cp.async.bulk.tensor is the preferred way.
            // This loop uses cp.async.bulk for pedagogical clarity:
            for (int i = 0; i < TILE_BYTES; i += 16) {
                uint32_t smem_off = __cvta_generic_to_shared(smem) + i;
                uint64_t gmem_addr = reinterpret_cast<uint64_t>(
                    reinterpret_cast<const char*>(src) + i);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], 16, [%2];"
                    : : "r"(smem_off), "l"(gmem_addr), "r"(mbar_addr)
                );
            }
            // Bulk loads complete through mbarrier, without a commit_group.
        }

        // STEP 4: All threads do independent work while copy runs.
        // In a real kernel: compute on previous tile's data here.
        // For this demo: just count to simulate useful work.
        volatile int dummy = tid * k;  // prevent optimizer from removing
        (void)dummy;

        // STEP 5: Poll mbarrier until the copy completes.
        // mbarrier.try_wait.parity:
        //   Returns true when internal_phase != phaseParity
        //   i.e., when the barrier has flipped past the expected phase.
        {
            uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);
            uint32_t done = 0;
            while (!done) {
                asm volatile(
                    "{\n\t"
                    "  .reg .pred p;\n\t"
                    // try_wait.parity.acquire: non-blocking check WITH acquire fence.
                    // Returns p=true when phase has flipped (copy complete).
                    "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
                    "    p, [%1], %2;\n\t"
                    "  selp.u32 %0, 1, 0, p;\n\t"
                    "}\n\t"
                    : "=r"(done)
                    : "r"(mbar_addr), "r"((uint32_t)phase)
                );
            }
        }

        // After try_wait returns, the acquire fence guarantees that all
        // data written by the async copy is visible to this thread.
        // phase has NOT yet been updated — do that after using the data.

        // STEP 6: Verify the loaded data is correct.
        // Expected: smem[i] = (float)(k * TILE_SIZE + i)
        int local_passes = 0;
        for (int i = tid; i < TILE_SIZE; i += num_threads) {
            float expected = (float)(k * TILE_SIZE + i);
            if (fabsf(smem[i] - expected) < 1e-4f) local_passes++;
        }
        // Accumulate to global (atomic to avoid races across threads)
        atomicAdd(pass_count, (uint32_t)local_passes);

        // STEP 7: Flip phase for next iteration (barrier reuse).
        // The barrier's thread arrival count automatically reset to 1
        // when the phase flipped. No need to re-init!
        phase ^= 1;

        // Sync all threads before next iteration (so producer doesn't
        // call arrive.expect_tx before all threads are done reading SMEM)
        __syncthreads();
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex01: mbarrier Pipeline — Init/Arrive/Wait/Reuse        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Check GPU supports mbarrier (requires sm_90a / Hopper)
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: mbarrier requires Hopper (SM 9.0+). "
                        "Your GPU is SM %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    const int K = 4;              // number of tiles to process
    const int TOTAL = K * TILE_SIZE;
    const int BLOCK_SIZE = 64;

    // Build host input: tile k has values k*256, k*256+1, ..., k*256+255
    float h_input[TOTAL];
    for (int i = 0; i < TOTAL; i++) h_input[i] = (float)i;

    float *d_input;
    uint32_t *d_passes, h_passes = 0;
    CHECK_CUDA(cudaMalloc(&d_input, TOTAL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_passes, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, TOTAL * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_passes, 0, sizeof(uint32_t)));

    printf("Config: K=%d tiles, %d elements/tile, %d threads/block\n\n",
           K, TILE_SIZE, BLOCK_SIZE);

    mbarrier_pipeline_kernel<<<1, BLOCK_SIZE>>>(d_input, d_passes, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_passes, d_passes, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    uint32_t expected_passes = (uint32_t)K * TILE_SIZE;
    printf("Element checks: %u / %u passed\n", h_passes, expected_passes);

    bool all_pass = (h_passes == expected_passes);
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", all_pass
           ? "✓ ALL PASS — mbarrier pipeline working correctly!"
           : "✗ FAILURES DETECTED");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_passes));
    return all_pass ? 0 : 1;
}
