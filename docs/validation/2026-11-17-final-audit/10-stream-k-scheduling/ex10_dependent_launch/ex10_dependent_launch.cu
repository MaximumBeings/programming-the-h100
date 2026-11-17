// ═══════════════════════════════════════════════════════════════════
// ex10_dependent_launch.cu
//
// PURPOSE: Demonstrate griddepcontrol dependent-launch protocol.
//
// SETUP:
//   Kernel A: producer — writes output[i] = 3*i+1
//             emits griddepcontrol.launch_dependents at end
//   Kernel B: consumer — waits at griddepcontrol.wait
//             reads output[i], adds 1000, writes result[i]
//
//   Host launches with cudaLaunchAttributeProgrammaticStreamSerialization
//   to authorize early overlap.
//
// WHAT TO OBSERVE:
//   - Both kernels produce correct results (no data races)
//   - Kernel B correctly reads Kernel A's output (no stale data)
//
// WORKS ON: sm_90+ (H100, B100)
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

constexpr int N = 256;  // array size

// ── Kernel A: Producer ────────────────────────────────────────────
// Writes output[i] = 3*i+1, then signals dependent launch.
__global__ void kernel_producer(int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 3 * idx + 1;
    }

    // All threads participate: the CTA signals after its work is done.
    // __threadfence() is a device-scope (.gpu) release fence: it's enough
    // here because the dependent grid (Kernel B) launches on this same
    // GPU, so only device-scope visibility is required before the
    // griddepcontrol signal propagates. (Section 9.2 teaches this same
    // .gpu-scope fence; reach for __threadfence_system() -- a .sys-scope
    // fence -- only when a CPU host thread, a peer GPU, or another
    // process must observe the write, which costs more and isn't needed
    // for an intra-GPU producer/consumer pair like this one.)
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();  // release fence: all stores committed, device scope
    }
    __syncthreads();

    // Signal: dependent grid may now be dispatched.
    // Only the FIRST invocation per CTA has effect.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// ── Kernel B: Consumer ────────────────────────────────────────────
// Waits for Kernel A to complete, then reads and increments.
__global__ void kernel_consumer(const int* __restrict__ output,
                                int* __restrict__ result, int n) {
    // Park here until all prerequisite grids have retired.
    // Prevents reading output[] before Kernel A has written it.
    asm volatile("griddepcontrol.wait;" ::: "memory");

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = output[idx] + 1000;
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  ex10_dependent_launch — GDC Dependent Grid Protocol\n");
    printf("  N=%d elements\n", N);
    printf("═══════════════════════════════════════════════════════\n\n");

    // Check sm_90 capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 9) {
        printf("WARNING: griddepcontrol requires sm_90+. "
               "This device is sm_%d%d. Running with normal serialization.\n",
               prop.major, prop.minor);
    }

    int* d_output;
    int* d_result;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output, 0, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_result, 0, N * sizeof(int)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int BLOCK_SIZE = 128;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── Launch Kernel A normally ──────────────────────────────────
    printf("Launching Kernel A (producer)...\n");
    kernel_producer<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_output, N);

    // ── Launch Kernel B with programmatic serialization ───────────
    printf("Launching Kernel B (consumer) with programmatic serialization...\n");

    cudaLaunchAttribute attr[1];
    attr[0].id  = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = dim3(num_blocks);
    cfg.blockDim = dim3(BLOCK_SIZE);
    cfg.dynamicSmemBytes = 0;
    cfg.stream   = stream;
    cfg.attrs    = attr;
    cfg.numAttrs = 1;

    CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel_consumer, d_output, d_result, N));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // ── Verify results ────────────────────────────────────────────
    int h_output[N], h_result[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nVerification (first 8 elements):\n");
    bool all_pass = true;
    for (int i = 0; i < N; ++i) {
        int expected_out    = 3 * i + 1;
        int expected_result = expected_out + 1000;
        bool out_ok    = (h_output[i] == expected_out);
        bool result_ok = (h_result[i] == expected_result);
        if (!out_ok || !result_ok) all_pass = false;
        if (i < 8) {
            printf("  [%3d] A wrote %4d (exp %4d) %s  "
                   "B result %4d (exp %4d) %s\n",
                   i,
                   h_output[i], expected_out,    out_ok    ? "✓" : "✗",
                   h_result[i], expected_result, result_ok ? "✓" : "✗");
        }
    }
    if (N > 8) printf("  ... (%d more elements)\n", N - 8);

    printf("\n%s\n", all_pass ? "ALL PASS ✓" : "FAILURES DETECTED ✗");

    cudaFree(d_output);
    cudaFree(d_result);
    cudaStreamDestroy(stream);
    return all_pass ? 0 : 1;
}
