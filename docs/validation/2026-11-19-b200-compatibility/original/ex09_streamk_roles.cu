// ═══════════════════════════════════════════════════════════════════
// ex09_streamk_roles.cu
//
// PURPOSE: Demonstrate Stream-K first/middle/final split logic.
//
// SIMULATED SETUP:
//   3 output tiles, each with K_TILES=90 k-tiles.
//   4 "virtual CTAs" (represented as CPU threads for clarity, then
//   mirrored as actual CUDA blocks for correctness verification).
//   Each k-tile contributes value (global_k_tile_idx + 1) to the sum.
//
// This kernel is educational: it uses simple integer accumulation
// instead of real matrix math so you can trace every number.
//
// WORKS ON: sm_80+
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cuda/atomic>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// ── Problem constants ─────────────────────────────────────────────
constexpr int NUM_OUTPUT_TILES     = 3;
constexpr int K_TILES_PER_OUTPUT   = 90;
constexpr int TOTAL_K_ITERS        = NUM_OUTPUT_TILES * K_TILES_PER_OUTPUT; // 270
constexpr int NUM_CTAS             = 4;

// ── Split descriptor ──────────────────────────────────────────────
struct Split {
    int tile_id;       // which output tile [0, NUM_OUTPUT_TILES)
    int K_idx;         // start within tile's K dimension
    int k_tile_count;  // how many k-tiles this split covers
};

// ── Helper: is_final_split ────────────────────────────────────────
__host__ __device__ bool is_final_split(const Split& s) {
    return (s.K_idx + s.k_tile_count) == K_TILES_PER_OUTPUT;
}

// ── Helper: requires_fixup ────────────────────────────────────────
__host__ __device__ bool requires_fixup(const Split& s) {
    return s.k_tile_count != K_TILES_PER_OUTPUT;
}

// ── Compute the value contributed by a single k-tile ──────────────
// Global k-tile index within the entire problem:
//   global_k = tile_id * K_TILES_PER_OUTPUT + local_k
// Value = global_k + 1  (so first k-tile contributes 1, last 270)
__host__ __device__ long long k_tile_value(int tile_id, int local_k) {
    int global_k = tile_id * K_TILES_PER_OUTPUT + local_k;
    return (long long)(global_k + 1);
}

// ── Accumulate k-tiles [K_idx, K_idx+k_tile_count) for a tile ────
__host__ __device__ long long accumulate_split(const Split& s) {
    long long sum = 0;
    for (int k = s.K_idx; k < s.K_idx + s.k_tile_count; ++k) {
        sum += k_tile_value(s.tile_id, k);
    }
    return sum;
}

// ── GPU kernel: each block handles one split ──────────────────────
// workspace[tile_id]:    partial sum storage (one long long per tile)
// lock[tile_id]:         atomic progress counter
// output[tile_id]:       final result written by the final split
__global__ void streamk_splits_kernel(
    const Split* __restrict__ splits,  // split descriptors, one per block
    long long*   workspace,            // partial sum buffer [NUM_OUTPUT_TILES]
    int*         lock,                 // progress counters  [NUM_OUTPUT_TILES]
    long long*   output)               // final results      [NUM_OUTPUT_TILES]
{
    int cta_id = blockIdx.x;
    const Split& s = splits[cta_id];

    // STEP 1: Accumulate this split's K-tiles locally
    long long my_partial = accumulate_split(s);

    // STEP 2: Apply split role logic
    if (!requires_fixup(s)) {
        // ── Full tile: this CTA owns all K-tiles ──────────────────
        // No coordination needed. Write directly to output.
        printf("  CTA %d → tile %d: FULL TILE, sum=%lld → write output\n",
               cta_id, s.tile_id, my_partial);
        output[s.tile_id] = my_partial;

    } else if (s.K_idx == 0) {
        // ── First split: write to workspace, signal lock ──────────
        printf("  CTA %d → tile %d: FIRST SPLIT [0,%d), partial=%lld\n",
               cta_id, s.tile_id, s.k_tile_count, my_partial);
        workspace[s.tile_id] = my_partial;
        __threadfence();  // ensure workspace visible before lock update
        atomicAdd(&lock[s.tile_id], s.k_tile_count);

    } else if (is_final_split(s)) {
        // ── Final split: wait, merge, write output ────────────────
        printf("  CTA %d → tile %d: FINAL SPLIT [%d,%d), waiting for lock>=%d\n",
               cta_id, s.tile_id, s.K_idx, s.K_idx + s.k_tile_count, s.K_idx);

        // Spin until all prior splits have written their sums
        while (atomicAdd(&lock[s.tile_id], 0) < s.K_idx) {
            __nanosleep(100);  // avoid L2 hammering during spin
        }
        __threadfence();  // ensure we see the workspace writes

        long long prior_sum = workspace[s.tile_id];
        long long total     = my_partial + prior_sum;

        printf("  CTA %d → tile %d: FINAL SPLIT merged: prior=%lld + mine=%lld = %lld\n",
               cta_id, s.tile_id, prior_sum, my_partial, total);

        // (Real kernel would run epilogue here: alpha*C + bias + relu etc.)
        output[s.tile_id] = total;

    } else {
        // ── Middle split: wait, merge, re-write workspace ─────────
        printf("  CTA %d → tile %d: MIDDLE SPLIT [%d,%d), waiting for lock>=%d\n",
               cta_id, s.tile_id, s.K_idx, s.K_idx + s.k_tile_count, s.K_idx);

        while (atomicAdd(&lock[s.tile_id], 0) < s.K_idx) {
            __nanosleep(100);
        }
        __threadfence();

        long long prior_sum = workspace[s.tile_id];
        long long merged    = my_partial + prior_sum;

        workspace[s.tile_id] = merged;
        __threadfence();
        atomicAdd(&lock[s.tile_id], s.k_tile_count);

        printf("  CTA %d → tile %d: MIDDLE SPLIT wrote merged=%lld back to workspace\n",
               cta_id, s.tile_id, merged);
    }
}

// ── CPU reference: compute expected output for each tile ──────────
long long cpu_reference(int tile_id) {
    long long sum = 0;
    for (int k = 0; k < K_TILES_PER_OUTPUT; ++k) {
        sum += k_tile_value(tile_id, k);
    }
    return sum;
}

// ── Build split descriptors for the 4-CTA scenario ────────────────
// Mirrors the worked example from Section 2.1.
void build_splits(Split* splits, int& num_splits) {
    // Linear assignment: 270 iters / 4 CTAs = 67 or 68 each
    // CTA 0: iters [0,   67)
    // CTA 1: iters [67, 134)
    // CTA 2: iters [134,202)
    // CTA 3: iters [202,270)
    int starts[NUM_CTAS+1] = {0, 67, 134, 202, 270};
    num_splits = 0;

    for (int cta = 0; cta < NUM_CTAS; ++cta) {
        int iter_start = starts[cta];
        int iter_end   = starts[cta+1];
        int iter_cur   = iter_start;

        while (iter_cur < iter_end) {
            // Which output tile does iter_cur fall in?
            int tile_id = iter_cur / K_TILES_PER_OUTPUT;
            // Start of this tile in global iter space
            int tile_iter_start = tile_id * K_TILES_PER_OUTPUT;
            // End of this tile in global iter space
            int tile_iter_end   = tile_iter_start + K_TILES_PER_OUTPUT;

            int split_end   = (iter_end < tile_iter_end) ? iter_end : tile_iter_end;
            int K_idx       = iter_cur - tile_iter_start;
            int k_tile_count = split_end - iter_cur;

            splits[num_splits++] = Split{tile_id, K_idx, k_tile_count};
            iter_cur = split_end;
        }
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  ex09_streamk_roles — Stream-K Split Role Demo\n");
    printf("  %d output tiles × %d K-tiles/tile = %d total iters\n",
           NUM_OUTPUT_TILES, K_TILES_PER_OUTPUT, TOTAL_K_ITERS);
    printf("  %d CTAs\n", NUM_CTAS);
    printf("═══════════════════════════════════════════════════════\n\n");

    // Build split descriptors
    Split h_splits[16];
    int num_splits = 0;
    build_splits(h_splits, num_splits);

    printf("Split assignment:\n");
    for (int i = 0; i < num_splits; ++i) {
        const Split& s = h_splits[i];
        const char* role = (s.K_idx == 0 && s.k_tile_count == K_TILES_PER_OUTPUT) ? "FULL" :
                           (s.K_idx == 0) ? "FIRST" :
                           is_final_split(s) ? "FINAL" : "MIDDLE";
        printf("  Split %d: tile=%d K_idx=%2d k_tile_count=%2d role=%s\n",
               i, s.tile_id, s.K_idx, s.k_tile_count, role);
    }
    printf("\n");

    // Allocate GPU buffers
    Split*     d_splits;
    long long* d_workspace;
    int*       d_lock;
    long long* d_output;

    CHECK_CUDA(cudaMalloc(&d_splits,    num_splits        * sizeof(Split)));
    CHECK_CUDA(cudaMalloc(&d_workspace, NUM_OUTPUT_TILES  * sizeof(long long)));
    CHECK_CUDA(cudaMalloc(&d_lock,      NUM_OUTPUT_TILES  * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output,    NUM_OUTPUT_TILES  * sizeof(long long)));

    CHECK_CUDA(cudaMemcpy(d_splits, h_splits, num_splits * sizeof(Split),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_workspace, 0, NUM_OUTPUT_TILES * sizeof(long long)));
    CHECK_CUDA(cudaMemset(d_lock,      0, NUM_OUTPUT_TILES * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output,    0, NUM_OUTPUT_TILES * sizeof(long long)));

    // Launch: one block per split
    printf("Kernel output:\n");
    streamk_splits_kernel<<<num_splits, 1>>>(d_splits, d_workspace, d_lock, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\n");

    // Copy results back and verify
    long long h_output[NUM_OUTPUT_TILES];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, NUM_OUTPUT_TILES * sizeof(long long),
                          cudaMemcpyDeviceToHost));

    printf("Verification:\n");
    bool all_pass = true;
    for (int t = 0; t < NUM_OUTPUT_TILES; ++t) {
        long long expected = cpu_reference(t);
        bool ok = (h_output[t] == expected);
        if (!ok) all_pass = false;
        printf("  Tile %d: got=%lld  expected=%lld  %s\n",
               t, h_output[t], expected, ok ? "PASS ✓" : "FAIL ✗");
    }

    printf("\n%s\n", all_pass ? "ALL PASS ✓" : "FAILURES DETECTED ✗");

    cudaFree(d_splits);
    cudaFree(d_workspace);
    cudaFree(d_lock);
    cudaFree(d_output);
    return all_pass ? 0 : 1;
}
