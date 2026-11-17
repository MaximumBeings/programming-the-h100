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
    float expected_10 = 280.0f + 0.08f * 10;  // j=10: sum = 280.8 (CORRECTED comment: 280.0+0.08*10=280.8, not 281)
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
