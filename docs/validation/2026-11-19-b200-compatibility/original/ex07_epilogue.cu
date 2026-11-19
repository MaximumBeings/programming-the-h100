// ═══════════════════════════════════════════════════════════════════
// ex07_epilogue.cu
//
// PURPOSE: Implement a full fused epilogue: α×acc + β×C + bias → ReLU.
//   Demonstrates register-resident computation (no intermediate SMEM),
//   bias broadcast, activation, and R2S→TMA output pattern.
//
// SIMPLIFICATION: Uses direct global stores instead of TMA for clarity.
//   A production kernel would insert the R→S→TMA path from §16.
//
// WORKS ON: sm_80+ (no Hopper-specific features)
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

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// ── Epilogue kernel ───────────────────────────────────────────────
// Computes: D[m][n] = ReLU( α×acc[m][n] + β×C[m][n] + bias[m] )
//
// Each thread handles one element of the tile.
// In a real WGMMA kernel, each thread handles MULTIPLE elements
// (the accumulator register fragment). Here, 1 thread = 1 element
// for clarity.
__global__ void epilogue_kernel(
    const float* acc,     // simulated WGMMA output (TILE_M × TILE_N)
    const float* C_mat,   // residual input matrix (TILE_M × TILE_N)
    const float* bias,    // per-row bias vector (TILE_M)
    float* D_out,         // fused output (TILE_M × TILE_N)
    float alpha,          // scaling for accumulator
    float beta,           // scaling for C matrix
    int M, int N
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // col index
    if (m >= M || n >= N) return;

    int idx = m * N + n;

    // ── STAGE 3 (previsit): Load inputs to registers ──────────────
    // In a real kernel, acc values are already IN registers from WGMMA.
    // C values are loaded from SMEM (pre-loaded by epilogue DMA warp).
    // Here we load from global memory to simulate the register state.
    float acc_val  = acc[idx];      // from WGMMA registers (simulated)
    float c_val    = C_mat[idx];    // from SMEM (loaded by epi DMA warp)
    float bias_val = bias[m];       // broadcast: same for all n in row m

    // ── STAGE 4 (visit): Arithmetic — ALL IN REGISTERS ───────────
    // Step 1: Linear combination  D = α×acc + β×C
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"  // result = alpha * acc_val + (beta * c_val)
        : "=f"(result)
        : "f"(alpha), "f"(acc_val), "f"(beta * c_val));
    // Note: beta*c_val computed first (compiler may optimize order).
    // In PTX: fma(alpha, acc_val, beta*c_val) = alpha*acc_val + beta*c_val

    // Step 2: Bias add (per-row broadcast — same bias for all n)
    result += bias_val;

    // Step 3: ReLU activation — register-level, zero cost for memory
    // PTX equivalent: max.f32 result, result, 0.0
    result = result > 0.0f ? result : 0.0f;
    // Or via PTX: asm("max.f32 %0, %0, 0f00000000;" : "+f"(result));

    // ── STAGE 6 (tma_store): Write output ────────────────────────
    // In production: R→S (stmatrix with swizzle) then TMA store.
    // Here: direct global store for demo simplicity.
    D_out[idx] = result;
}

// ── FP8 amax reduction demo ───────────────────────────────────────
// Demonstrates the amax computation needed for FP8 epilogue.
// Uses warp shuffle for intra-warp reduction.
__global__ void compute_tile_amax(
    const float* tile,    // TILE_M × TILE_N values
    float* global_amax,   // output: max(|tile|)
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    // ── STAGE 5 (reduce): Compute amax ────────────────────────────
    // Each thread tracks its local amax
    float local_amax = 0.0f;
    if (idx < total) {
        local_amax = fabsf(tile[idx]);
    }

    // Warp-level shuffle reduction: reduce within 32 threads of a warp
    // __reduce_max_sync is available from CUDA 8.0+
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, local_amax, offset);
        local_amax = fmaxf(local_amax, other);
    }

    // Lane 0 of each warp writes to shared memory
    __shared__ float warp_amax[32];  // up to 32 warps per block
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) warp_amax[warp] = local_amax;
    __syncthreads();

    // Thread 0 reduces across all warps
    if (threadIdx.x == 0) {
        float block_amax = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            block_amax = fmaxf(block_amax, warp_amax[w]);
        }
        // Atomic max across blocks
        // Note: CUDA has no atomicMax for float, use bit trick:
        // (Works because IEEE 754 positive floats sort like integers)
        atomicMax((int*)global_amax, __float_as_int(block_amax));
    }
}

// ─── Main ──────────────────────────────────────────────────────────
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Epilogue Fusion Demo                                 ║\n");
    printf("║  D = ReLU( α×acc + β×C + bias[row] )                 ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    const float ALPHA = 1.0f;
    const float BETA  = 0.5f;

    // ── Generate test data on host ────────────────────────────────
    float h_acc[TILE_M][TILE_N], h_C[TILE_M][TILE_N];
    float h_bias[TILE_M], h_D[TILE_M][TILE_N], h_expected[TILE_M][TILE_N];

    for (int m = 0; m < TILE_M; m++) {
        h_bias[m] = m * 0.1f;  // bias[m] = m/10
        for (int n = 0; n < TILE_N; n++) {
            // acc: mix of positive and negative
            h_acc[m][n] = (m - TILE_M/2) * 1.0f + (n % 7) * 0.3f - 2.0f;
            h_C[m][n]   = (n - TILE_N/2) * 0.5f;
        }
    }

    // Compute expected output on host (for verification)
    int neg_count = 0;
    for (int m = 0; m < TILE_M; m++) {
        for (int n = 0; n < TILE_N; n++) {
            float val = ALPHA * h_acc[m][n] + BETA * h_C[m][n] + h_bias[m];
            h_expected[m][n] = fmaxf(0.0f, val);  // ReLU
            if (val < 0.0f) neg_count++;
        }
    }

    printf("Input tile: %d×%d, alpha=%.1f, beta=%.1f\n", TILE_M, TILE_N, ALPHA, BETA);
    printf("Bias: bias[m] = m * 0.1\n");
    printf("Expected ReLU clamped to 0: %d/%d elements\n\n",
           neg_count, TILE_M * TILE_N);

    // ── Allocate device memory ────────────────────────────────────
    float *d_acc, *d_C, *d_bias, *d_D, *d_amax;
    size_t tile_bytes = TILE_M * TILE_N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_acc,  tile_bytes));
    CHECK_CUDA(cudaMalloc(&d_C,    tile_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, TILE_M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D,    tile_bytes));
    CHECK_CUDA(cudaMalloc(&d_amax, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_acc,  h_acc,  tile_bytes,            cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C,    h_C,    tile_bytes,            cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, TILE_M*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_D,    0, tile_bytes));
    CHECK_CUDA(cudaMemset(d_amax, 0, sizeof(float)));

    // ── Launch epilogue kernel ────────────────────────────────────
    dim3 block(16, 16);  // 256 threads
    dim3 grid((TILE_N + 15) / 16, (TILE_M + 15) / 16);
    epilogue_kernel<<<grid, block>>>(d_acc, d_C, d_bias, d_D, ALPHA, BETA, TILE_M, TILE_N);
    CHECK_CUDA(cudaGetLastError());

    // ── Launch amax kernel ────────────────────────────────────────
    compute_tile_amax<<<(TILE_M*TILE_N+255)/256, 256>>>(d_D, d_amax, TILE_M, TILE_N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ── Copy results back ─────────────────────────────────────────
    CHECK_CUDA(cudaMemcpy(h_D,    d_D,    tile_bytes, cudaMemcpyDeviceToHost));
    float h_amax;
    CHECK_CUDA(cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

    // ── Verify ───────────────────────────────────────────────────
    int errors = 0;
    float max_err = 0.0f;
    for (int m = 0; m < TILE_M; m++) {
        for (int n = 0; n < TILE_N; n++) {
            float err = fabsf(h_D[m][n] - h_expected[m][n]);
            max_err = fmaxf(max_err, err);
            if (err > 1e-4f) {
                errors++;
                if (errors <= 3)
                    printf("  FAIL at [%d][%d]: got %.5f expected %.5f\n",
                           m, n, h_D[m][n], h_expected[m][n]);
            }
        }
    }

    printf("Verification:\n");
    printf("  Total elements: %d\n", TILE_M * TILE_N);
    printf("  Errors: %d\n", errors);
    printf("  Max absolute error: %.6f (threshold: 1e-4)\n", max_err);
    printf("\nSample outputs:\n");
    printf("  D[0][0] = %.4f (expected %.4f)\n", h_D[0][0], h_expected[0][0]);
    printf("  D[2][5] = %.4f (expected %.4f)\n", h_D[2][5], h_expected[2][5]);
    printf("  D[3][2] = %.4f (expected %.4f, should be 0 if pre-ReLU < 0)\n",
           h_D[3][2], h_expected[3][2]);

    printf("\nFP8 amax reduction result: %.4f\n", h_amax);
    printf("  (This is max|D[m][n]| across the tile, used for FP8 scaling)\n");

    bool all_pass = (errors == 0);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (all_pass)
        printf("║  ✓ ALL %d ELEMENTS CORRECT — epilogue fusion works!   ║\n",
               TILE_M * TILE_N);
    else
        printf("║  ✗ %d FAILURES — check output above                  ║\n", errors);
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_acc)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_bias)); CHECK_CUDA(cudaFree(d_D)); CHECK_CUDA(cudaFree(d_amax));
    return all_pass ? 0 : 1;
}
