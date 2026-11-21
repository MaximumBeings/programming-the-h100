// ═══════════════════════════════════════════════════════════════════
// ex03_setmaxnreg.cu
//
// PURPOSE: Demonstrate setmaxnreg — dynamic register resizing.
//
// DESIGN:
//   WG0 (threads 0-127):   calls setmaxnreg.dec(32) → producer registers
//   WG1 (threads 128-255): calls setmaxnreg.inc(128) → consumer registers
//
//   After resizing, both warpgroups perform arithmetic to prove
//   that register state is preserved through the resize operation.
//
// IMPORTANT: setmaxnreg MUST be aligned (all threads in warpgroup
// must execute it simultaneously, hence .aligned variant).
//
// REQUIRES: sm_90a (Hopper) or sm_100a (Blackwell)
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
constexpr int PRODUCER_REGS  = 32;   // minimum for producer (TMA + barriers)
constexpr int CONSUMER_REGS  = 128;  // moderate consumer register budget

// ── Kernel ────────────────────────────────────────────────────────
// Tests that setmaxnreg works correctly and doesn't corrupt computation.
//
// Protocol:
//   1. All threads compute an initial value (before setmaxnreg)
//   2. setmaxnreg changes register allocation
//   3. All threads perform MORE arithmetic (uses same registers!)
//   4. We verify that the final result is correct
//
// If setmaxnreg corrupted registers, the second computation would be wrong.
__global__ void __maxnreg__(80) test_setmaxnreg(float* output_wg0, float* output_wg1) {
    int warpgroup_id = threadIdx.x / WARPGROUP_SIZE;
    int local_idx    = threadIdx.x % WARPGROUP_SIZE;

    // ── Step 1: Compute initial value (before any register resize) ─
    // Both warpgroups do this. The value lives in a register.
    float accumulator = (float)(local_idx * 2 + 1);  // 1, 3, 5, 7, ...
    float factor      = (float)(warpgroup_id + 1);    // WG0: 1.0, WG1: 2.0

    if (warpgroup_id == 0) {
        // ── PRODUCER warpgroup: decrease registers ────────────────
        // setmaxnreg.dec: "I am releasing registers down to PRODUCER_REGS"
        // The .sync means: this thread can't issue instructions after
        //                  this point until ALL threads in the warpgroup reach it.
        // The .aligned means: every thread in the warp group must
        //                     execute this together.
        asm volatile(
            "setmaxnreg.dec.sync.aligned.u32 %0;"
            :: "n"(PRODUCER_REGS)
        );
        // After this point: WG0 has PRODUCER_REGS registers available.
        // The register 'accumulator' and 'factor' STILL EXIST because
        // the compiler arranged them to fit within PRODUCER_REGS.
        // (If they didn't fit, the compiler would spill — check with --ptxas-options=-v)

    } else {
        // ── CONSUMER warpgroup: increase registers ─────────────────
        // setmaxnreg.inc: "I am claiming more registers up to CONSUMER_REGS"
        asm volatile(
            "setmaxnreg.inc.sync.aligned.u32 %0;"
            :: "n"(CONSUMER_REGS)
        );
        // After this point: WG1 has CONSUMER_REGS registers available.
        // The existing registers 'accumulator' and 'factor' are still valid.
        // WG1 now has room for more intermediate results.
    }

    // ── Step 2: More computation AFTER register resize ────────────
    // This tests that setmaxnreg did not corrupt the existing register state.
    // If registers were corrupted, this computation would produce wrong results.
    //
    // We do a small computation that uses the pre-resize values:
    //   result = accumulator * factor + local_idx * 0.5
    //
    // WG0: accumulator = 2*idx+1, factor = 1.0 → result = 2*idx+1 + 0.5*idx = 2.5*idx+1
    // WG1: accumulator = 2*idx+1, factor = 2.0 → result = 2*(2*idx+1) + 0.5*idx = 4.5*idx+2
    float result = accumulator * factor + (float)local_idx * 0.5f;

    // Do a few more ops to stress-test register integrity
    result = result * 1.0f + 0.0f;  // identity: should not change result
    result = result + 0.0f;

    // ── Step 3: Write results ──────────────────────────────────────
    if (warpgroup_id == 0) {
        output_wg0[local_idx] = result;
    } else {
        output_wg1[local_idx] = result;
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  setmaxnreg Demo — Dynamic Register Resizing         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Check for Hopper
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: setmaxnreg requires Hopper (SM 9.0+) or Blackwell.\n");
        return 1;
    }

    printf("Register Plan:\n");
    printf("  WG0 (threads   0-127): setmaxnreg.dec → %d regs (producer)\n",
           PRODUCER_REGS);
    printf("  WG1 (threads 128-255): setmaxnreg.inc → %d regs (consumer)\n\n",
           CONSUMER_REGS);

    float *d_wg0, *d_wg1;
    CHECK_CUDA(cudaMalloc(&d_wg0, WARPGROUP_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wg1, WARPGROUP_SIZE * sizeof(float)));

    test_setmaxnreg<<<1, 256>>>(d_wg0, d_wg1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_wg0[WARPGROUP_SIZE], h_wg1[WARPGROUP_SIZE];
    CHECK_CUDA(cudaMemcpy(h_wg0, d_wg0, WARPGROUP_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_wg1, d_wg1, WARPGROUP_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    // Verify WG0 results: result = (2*idx+1)*1.0 + idx*0.5 = 2.5*idx + 1
    printf("WG0 (PRODUCER, %d regs after resize):\n", PRODUCER_REGS);
    int errors_wg0 = 0, errors_wg1 = 0;
    for (int i = 0; i < WARPGROUP_SIZE; i++) {
        float expected = (2.0f * i + 1.0f) * 1.0f + (float)i * 0.5f;
        if (fabsf(h_wg0[i] - expected) > 1e-4f) {
            errors_wg0++;
            if (errors_wg0 <= 3)
                printf("  FAIL [%d]: got %.4f expected %.4f\n", i, h_wg0[i], expected);
        }
    }
    if (errors_wg0 == 0) {
        printf("  ✓ All %d values correct (registers NOT corrupted by setmaxnreg)\n",
               WARPGROUP_SIZE);
        printf("  Sample: output[0]=%.1f (expect 1.0), output[1]=%.1f (expect 3.5)\n",
               h_wg0[0], h_wg0[1]);
    }

    // Verify WG1 results: result = (2*idx+1)*2.0 + idx*0.5 = 4.5*idx + 2
    printf("\nWG1 (CONSUMER, %d regs after resize):\n", CONSUMER_REGS);
    for (int i = 0; i < WARPGROUP_SIZE; i++) {
        float expected = (2.0f * i + 1.0f) * 2.0f + (float)i * 0.5f;
        if (fabsf(h_wg1[i] - expected) > 1e-4f) {
            errors_wg1++;
            if (errors_wg1 <= 3)
                printf("  FAIL [%d]: got %.4f expected %.4f\n", i, h_wg1[i], expected);
        }
    }
    if (errors_wg1 == 0) {
        printf("  ✓ All %d values correct (registers NOT corrupted by setmaxnreg)\n",
               WARPGROUP_SIZE);
        printf("  Sample: output[0]=%.1f (expect 2.0), output[1]=%.1f (expect 6.5)\n",
               h_wg1[0], h_wg1[1]);
    }

    bool all_pass = (errors_wg0 == 0 && errors_wg1 == 0);
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (all_pass) {
        printf("║  ✓ ALL TESTS PASSED                                  ║\n");
        printf("║  setmaxnreg correctly resized registers without      ║\n");
        printf("║  corrupting existing register state.                 ║\n");
    } else {
        printf("║  ✗ FAILURES: WG0=%d errors, WG1=%d errors            ║\n",
               errors_wg0, errors_wg1);
    }
    printf("╚══════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_wg0)); CHECK_CUDA(cudaFree(d_wg1));
    return all_pass ? 0 : 1;
}
