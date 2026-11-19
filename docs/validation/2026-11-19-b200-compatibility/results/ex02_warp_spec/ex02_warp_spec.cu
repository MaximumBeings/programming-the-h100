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
