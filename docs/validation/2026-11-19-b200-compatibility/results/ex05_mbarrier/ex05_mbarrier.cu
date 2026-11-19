// ═══════════════════════════════════════════════════════════════════
// ex05_mbarrier.cu
//
// PURPOSE: Demonstrate the mbarrier two-barrier handshake between
//   a producer warpgroup and a consumer warpgroup.
//
// LAYOUT:
//   128 threads, 2 warpgroups, 1 pipeline stage, 3 rounds.
//   WG0 (producer): writes SMEM → signals FULL barrier.
//   WG1 (consumer): waits FULL barrier → reads SMEM → signals EMPTY.
//
// This is a simplified version of the real TMA pipeline.
// Replace the producer's manual SMEM write with a TMA cp.async.bulk
// to get the actual production pattern.
//
// REQUIRES: sm_90a or sm_100a
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
constexpr int N_ROUNDS = 3;   // number of producer-consumer handshakes

// ── mbarrier helpers ─────────────────────────────────────────────
// These inline functions wrap the raw PTX mbarrier instructions.
// They are not part of the CUDA standard library — you define them
// yourself based on the PTX specification.

// Initialize a barrier expecting `count` arrivals before it fires.
// Must be called by thread 0 before any thread uses the barrier.
__device__ void mbar_init(uint64_t* bar, int count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count) // runtime arrival count
    );
}

// Arrive at a barrier: signal that this thread's work is done.
// The barrier fires when total arrivals == the expected count.
__device__ void mbar_arrive(uint64_t* bar) {
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar))
    );
}

// Wait until barrier fires (i.e., expected count of arrivals reached).
// `phase` alternates between 0 and 1 on each barrier cycle.
// This prevents a thread from accidentally "seeing" the PREVIOUS cycle.
__device__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    // try_wait returns 1 if the barrier has fired for the given phase.
    // We spin until it fires. In production, the hardware has a
    // more efficient "wait" instruction, but try_wait + spin works.
    uint32_t bar_addr = __cvta_generic_to_shared(bar);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "WAIT_LOOP:\n\t"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "  @!p bra WAIT_LOOP;\n\t"  // if not done, loop back
        "}"
        :: "r"(bar_addr), "r"(phase)
    );
}

// ── Kernel ───────────────────────────────────────────────────────
__global__ void mbarrier_handshake(
    const float* prod_values,  // values for producer to write (N_ROUNDS)
    float* cons_output,        // output from consumer (N_ROUNDS)
    int*   sequence_log        // log of which round happened when (debug)
)
{
    // ── Shared memory ─────────────────────────────────────────────
    // One pipeline stage: a single float slot.
    // In a real kernel, this would be a full tile (32KB+).
    __shared__ float stage_data;          // the pipeline stage buffer

    // Two barriers per stage
    // full_bar: fires when producer has written data
    // empty_bar: fires when consumer has finished reading
    __shared__ uint64_t full_bar;         // "data is ready for consumer"
    __shared__ uint64_t empty_bar;        // "stage is free for producer"

    int warpgroup_id = threadIdx.x / WARPGROUP_SIZE;
    int local_idx    = threadIdx.x % WARPGROUP_SIZE;

    // ── Initialize barriers (thread 0 only) ───────────────────────
    // full_bar:  initialized to expect 1 arrival (producer arrives after write)
    // empty_bar: initialized to expect 1 arrival (consumer arrives after read)
    //            BUT: we pre-signal empty_bar so producer can start immediately
    if (threadIdx.x == 0) {
        mbar_init(&full_bar,  1);  // expects 1 arrival from producer
        mbar_init(&empty_bar, 1);  // expects 1 arrival from consumer
        // Pre-signal empty_bar: "stage is initially empty (writable)"
        // Without this, producer would wait forever on empty_bar round 0.
        mbar_arrive(&empty_bar);
    }
    __syncthreads();  // all threads must see initialized barriers before proceeding

    // ── Pipeline rounds ───────────────────────────────────────────
    uint32_t full_phase  = 0;   // alternates 0→1→0→1... after each barrier cycle
    uint32_t empty_phase = 0;   // tracks which "generation" of empty_bar we expect

    for (int round = 0; round < N_ROUNDS; round++) {
        if (warpgroup_id == 0) {
            // ── PRODUCER path ───────────────────────────────────
            // Step 1: Wait for stage to be EMPTY (consumer done reading)
            if (local_idx == 0) {
                mbar_wait(&empty_bar, empty_phase);
            }
            __syncwarp();  // all producer threads see the barrier fired

            // Step 2: Write to the stage (simulates TMA load)
            // Only thread 0 writes for simplicity; in TMA the hardware writes.
            if (local_idx == 0) {
                stage_data = prod_values[round];
            }
            // Memory fence: ensure write is visible before we signal consumer
            __threadfence_block();

            // Step 3: Signal FULL barrier ("data is ready")
            if (local_idx == 0) {
                mbar_arrive(&full_bar);
            }
            // empty_phase flips every time empty_bar completes a cycle
            // This is needed to correctly identify the "next" empty event
            if (local_idx == 0) empty_phase ^= 1;

        } else {
            // ── CONSUMER path ───────────────────────────────────
            // Step 1: Wait for stage to be FULL (producer done writing)
            if (local_idx == 0) {
                mbar_wait(&full_bar, full_phase);
            }
            __syncwarp();  // all consumer threads see barrier fired

            // Step 2: Read from stage and process
            // In a real kernel: issue WGMMA using the stage data
            if (local_idx == 0) {
                float val = stage_data;
                cons_output[round] = val * 2.0f;  // process: double the value
            }
            // Memory fence: ensure reads complete before we signal empty
            __threadfence_block();

            // Step 3: Signal EMPTY ("stage is free for reuse")
            if (local_idx == 0) {
                mbar_arrive(&empty_bar);
            }
            // full_phase flips every time full_bar completes a cycle
            if (local_idx == 0) full_phase ^= 1;
        }

        // Log: record the round (sequenced by a global __syncthreads)
        __syncthreads();  // wait for both WGs to complete this round
        if (threadIdx.x == 0) sequence_log[round] = round;
    }
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  mbarrier Two-Barrier Handshake Demo                  ║\n");
    printf("║  %d rounds of producer→consumer using full/empty bars  ║\n", N_ROUNDS);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Check for Hopper
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: mbarrier requires Hopper (SM 9.0+).\n");
        return 1;
    }

    // Producer values: what the producer writes each round
    float h_prod_vals[N_ROUNDS] = {42.0f, 99.5f, 7.25f};
    float h_cons_out[N_ROUNDS]  = {};
    int   h_seq[N_ROUNDS]       = {};

    float *d_prod, *d_cons;
    int*   d_seq;
    CHECK_CUDA(cudaMalloc(&d_prod, N_ROUNDS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cons, N_ROUNDS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_seq,  N_ROUNDS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_prod, h_prod_vals, N_ROUNDS*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_cons, 0, N_ROUNDS * sizeof(float)));

    printf("Producer values: [");
    for (int i = 0; i < N_ROUNDS; i++) printf("%.2f%s", h_prod_vals[i], i<N_ROUNDS-1?", ":"");
    printf("]\n");
    printf("Expected consumer outputs: [");
    for (int i = 0; i < N_ROUNDS; i++) printf("%.2f%s", h_prod_vals[i]*2.0f, i<N_ROUNDS-1?", ":"");
    printf("]\n\n");

    // 128 threads = 1 warpgroup for producer + 1 for consumer
    mbarrier_handshake<<<1, 256>>>(d_prod, d_cons, d_seq);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_cons_out, d_cons, N_ROUNDS*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_seq,      d_seq,  N_ROUNDS*sizeof(int),   cudaMemcpyDeviceToHost));

    printf("Handshake Results:\n");
    printf("  %-8s %-12s %-12s %-12s %-8s\n", "Round", "Produced", "Expected", "Got", "OK?");
    printf("  %s\n", "────────────────────────────────────────────────");

    int errors = 0;
    for (int i = 0; i < N_ROUNDS; i++) {
        float expected = h_prod_vals[i] * 2.0f;
        bool ok = fabsf(h_cons_out[i] - expected) < 1e-4f;
        if (!ok) errors++;
        printf("  %-8d %-12.2f %-12.2f %-12.2f %s\n",
               i, h_prod_vals[i], expected, h_cons_out[i], ok ? "✓ PASS" : "✗ FAIL");
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (errors == 0) {
        printf("║  ✓ ALL %d ROUNDS PASSED                               ║\n", N_ROUNDS);
        printf("║                                                      ║\n");
        printf("║  WHAT HAPPENED:                                      ║\n");
        printf("║  1. empty_bar pre-signaled → producer starts freely  ║\n");
        printf("║  2. Producer writes SMEM → arrives on full_bar       ║\n");
        printf("║  3. Consumer waits full_bar → reads SMEM → doubles   ║\n");
        printf("║  4. Consumer arrives on empty_bar → stage recyclable  ║\n");
        printf("║  5. phase bits prevent stale barrier reads           ║\n");
    } else {
        printf("║  ✗ %d/%d rounds FAILED — check output above           ║\n",
               errors, N_ROUNDS);
    }
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_prod)); CHECK_CUDA(cudaFree(d_cons)); CHECK_CUDA(cudaFree(d_seq));
    return errors == 0 ? 0 : 1;
}
