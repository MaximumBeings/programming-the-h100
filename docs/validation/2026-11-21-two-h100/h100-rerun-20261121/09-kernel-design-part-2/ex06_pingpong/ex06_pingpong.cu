// ═══════════════════════════════════════════════════════════════════
// ex06_pingpong.cu
//
// PURPOSE: Check even/odd tile assignment on the GPU.
//   Consumers compute numerical values for their assigned tiles.
//
// SCOPE: No ordered barrier, WGMMA or measured overlap is implemented.
//   This is not a concurrent pipeline protocol test. A production
//   kernel needs explicit ready/release and phase-order barriers.
//
// WORKS ON: sm_80+ (uses only basic CUDA features for simulation)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <atomic>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

constexpr int N_TILES  = 6;   // total output tiles
constexpr int K_TILES  = 4;   // k-tiles per output tile (inner loop depth)
constexpr int N_STAGES = 3;   // pipeline stages (circular buffer slots)

// ── Shared state (for ordered execution logging) ──────────────────
// In a real kernel, ordering is enforced by mbarrier.
// Here we use host-visible global arrays for verification logging.
struct EventLog {
    int tile_id;
    int consumer_id;  // 0 = C0, 1 = C1
    int event_type;   // 0 = MMA start, 1 = MMA done, 2 = Epi done
    int k_tile;       // which k-tile for MMA events
};

// ── Simulated MMA (fake: just accumulation by adding k*100+tile) ──
__device__ float fake_mma(int tile_id, int k_tile) {
    return (float)(k_tile * 100 + tile_id);
}

// ── Simulated epilogue (scale by 2) ──────────────────────────────
__device__ float fake_epilogue(float accumulated) {
    return accumulated * 2.0f;
}

// ── Main ping-pong kernel ─────────────────────────────────────────
__global__ void pingpong_kernel(
    float* tile_outputs,   // output[N_TILES]: final value per tile
    int*   log_consumer,   // which consumer handled each tile
    int*   overlap_check   // 1 if overlap detected (MMA+Epi simultaneous)
) {
    // ── Identify role ──────────────────────────────────────────────
    constexpr int WARPGROUP_SIZE = 128;
    int wg_id = threadIdx.x / WARPGROUP_SIZE;
    int local_idx = threadIdx.x % WARPGROUP_SIZE;

    // WG0 = producer (simulated: just coordinates)
    // WG1 = Consumer 0 (even tiles: 0, 2, 4)
    // WG2 = Consumer 1 (odd  tiles: 1, 3, 5)
    if (wg_id == 0) {
        // Producer: just waits here (would do TMA in production)
        return;
    }

    int consumer_id = wg_id - 1;  // 0 for WG1, 1 for WG2

    // ── Consumer loop: process every other tile ───────────────────
    for (int tile = consumer_id; tile < N_TILES; tile += 2) {
        if (local_idx != 0) continue;  // only thread 0 per WG for demo

        // ── MMA phase: iterate over K tiles ──────────────────────
        float accumulator = 0.0f;
        for (int k = 0; k < K_TILES; k++) {
            accumulator += fake_mma(tile, k);
        }

        // ── Epilogue phase: apply output transform ────────────────
        float output_val = fake_epilogue(accumulator);

        // Write output
        tile_outputs[tile] = output_val;
        log_consumer[tile] = consumer_id;
    }

    // Note: in a real ping-pong kernel, the MMA and epilogue phases
    // are separated by the ordered barrier. Consumer 0's epilogue for
    // tile 0 runs at the SAME TIME as Consumer 1's MMA for tile 1.
    // Here, we execute them sequentially per thread but verify the
    // OUTPUT to confirm correctness of the tile assignment logic.
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Ping-Pong Pipeline Simulation                        ║\n");
    printf("║  %d tiles, C0=even, C1=odd, K_TILES=%d per tile      ║\n",
           N_TILES, K_TILES);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    float *d_outputs;
    int   *d_log, *d_overlap;
    CHECK_CUDA(cudaMalloc(&d_outputs, N_TILES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_log,     N_TILES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_overlap, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_outputs, 0, N_TILES * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_overlap, 0, sizeof(int)));

    // 384 threads = WG0 (prod) + WG1 (C0) + WG2 (C1)
    pingpong_kernel<<<1, 384>>>(d_outputs, d_log, d_overlap);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_out[N_TILES];
    int   h_log[N_TILES];
    CHECK_CUDA(cudaMemcpy(h_out, d_outputs, N_TILES*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_log, d_log,     N_TILES*sizeof(int),   cudaMemcpyDeviceToHost));

    printf("Tile Assignment and Output Verification:\n");
    printf("  MMA formula: accumulate k*100+tile_id for k=0..%d-1\n", K_TILES);
    printf("  sum per tile = sum(k=0..%d of k*100) + tile_id × %d\n",
           K_TILES-1, K_TILES);
    printf("  = %d + tile_id × %d\n", K_TILES*(K_TILES-1)/2*100, K_TILES);
    printf("  After epilogue (×2):\n\n");

    printf("  %-6s %-10s %-12s %-12s %-12s\n",
           "Tile", "Consumer", "Expected", "Got", "OK?");
    printf("  %s\n", "──────────────────────────────────────────────");

    // Expected: sum of k*100+tile for k=0..K_TILES-1 = 600+4*tile (for K_TILES=4)
    // Then × 2 for epilogue
    int errors = 0;
    int correct_consumer = 0;
    for (int t = 0; t < N_TILES; t++) {
        float acc = 0.0f;
        for (int k = 0; k < K_TILES; k++) acc += k * 100.0f + t;
        float expected = acc * 2.0f;
        int expected_consumer = t % 2;

        bool val_ok = fabsf(h_out[t] - expected) < 0.5f;
        bool con_ok = h_log[t] == expected_consumer;
        if (!val_ok || !con_ok) errors++;
        if (con_ok) correct_consumer++;

        printf("  %-6d C%-9d %-12.0f %-12.0f %s\n",
               t, h_log[t], expected, h_out[t],
               (val_ok && con_ok) ? "✓ PASS" : "✗ FAIL");
    }

    printf("\nPing-Pong Behavior Check:\n");
    printf("  Even tiles processed by Consumer 0: ");
    bool even_ok = true;
    for (int t = 0; t < N_TILES; t += 2) {
        if (h_log[t] != 0) { even_ok = false; printf("FAIL at t=%d ", t); }
    }
    printf("%s\n", even_ok ? "✓ ALL CORRECT" : "✗ ERROR");

    printf("  Odd tiles processed by Consumer 1:  ");
    bool odd_ok = true;
    for (int t = 1; t < N_TILES; t += 2) {
        if (h_log[t] != 1) { odd_ok = false; printf("FAIL at t=%d ", t); }
    }
    printf("%s\n", odd_ok ? "✓ ALL CORRECT" : "✗ ERROR");

    bool all_pass = (errors == 0 && even_ok && odd_ok);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (all_pass) {
        printf("║  ✓ ALL TESTS PASSED                                  ║\n");
        printf("║                                                      ║\n");
        printf("║  In production, C0's epilogue overlaps C1's MMA:     ║\n");
        printf("║  C0: [MMA T0][EPI T0][MMA T2][EPI T2]               ║\n");
        printf("║  C1:    [MMA T1][EPI T1][MMA T3][EPI T3]            ║\n");
        printf("║  Tile assignment checked; no Tensor Core utilization or overlap measured          ║\n");
    } else {
        printf("║  ✗ %d FAILURES — check output above                  ║\n", errors);
    }
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_outputs));
    CHECK_CUDA(cudaFree(d_log));
    CHECK_CUDA(cudaFree(d_overlap));
    return all_pass ? 0 : 1;
}
