// ═══════════════════════════════════════════════════════════════════
// ex02_war_prevention.cu
//
// PURPOSE: Demonstrate WAR hazard prevention using two mbarriers.
//
//   data_bar:    "Has the data arrived from the async copy?"
//                Producer signals (via TMA hardware). Consumers wait.
//
//   release_bar: "Are consumers done reading the buffer?"
//                Consumers signal (via mbarrier.arrive).
//                Producer waits before overwriting the buffer.
//
// PIPELINE:
//   Producer thread: copies tile K into the current buffer
//   Consumer threads: compute on the current buffer
//   Neither overwrites until the other is done.
//
// REQUIRES: sm_90a (H100) or sm_100a (Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr,"CUDA ERROR %s:%d: %s\n", \
                __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

static constexpr int TILE   = 128;   // elements per tile
static constexpr int TBYTES = TILE * sizeof(float);
static constexpr int K_TILES = 8;   // total tiles

__global__ void war_prevention_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int K)
{
    // ── Shared memory ─────────────────────────────────────────────
    // Two barriers: data_bar (copy→compute handoff) and
    //               release_bar (compute→copy handoff, WAR prevention)
    __shared__ __align__(8) uint64_t data_bar;
    __shared__ __align__(8) uint64_t release_bar;
    __shared__ __align__(16) float smem[TILE];    // single buffer (simplified)

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    uint32_t data_addr    = __cvta_generic_to_shared(&data_bar);
    uint32_t release_addr = __cvta_generic_to_shared(&release_bar);

    // ── INIT (thread 0 only, then syncthreads) ────────────────────
    if (tid == 0) {
        // data_bar: 1 producer thread will call arrive.expect_tx
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(data_addr));
        // release_bar: ALL consumer threads will call arrive when done reading
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     : : "r"(release_addr), "r"(nthreads));
    }
    // Publish initialized barrier state to the bulk-async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    int data_phase    = 0;   // phase for data_bar
    int release_phase = 0;   // phase for release_bar

    for (int k = 0; k < K; k++) {

        // ── PRODUCER: wait for consumers to release (WAR prevention) ─
        // On first iteration: release_bar was just initialized but no one
        // has arrived yet, so we pre-arrive all threads to "open" it.
        if (k == 0) {
            // Prime release_bar: all threads arrive so it fires immediately
            // (no previous computation, so the buffer is "free" to write)
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                : : "r"(release_addr)
            );
        }

        // Producer thread waits on release_bar before overwriting SMEM
        if (tid == 0) {
            // Spin until consumers signal "buffer is free to overwrite"
            uint32_t ready = 0;
            while (!ready) {
                asm volatile(
                    "{\n\t"
                    "  .reg .pred p;\n\t"
                    "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
                    "    p, [%1], %2;\n\t"
                    "  selp.u32 %0, 1, 0, p;\n\t"
                    "}\n\t"
                    : "=r"(ready) : "r"(release_addr), "r"((uint32_t)release_phase)
                );
            }
            release_phase ^= 1;

            // Now safe to load next tile into SMEM
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                : : "r"(data_addr), "r"((uint32_t)TBYTES)
            );

            const float* tile_src = src + k * TILE;
            for (int i = 0; i < TBYTES; i += 16) {
                uint32_t s = __cvta_generic_to_shared(smem) + i;
                uint64_t g = reinterpret_cast<uint64_t>(
                    reinterpret_cast<const char*>(tile_src) + i);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], 16, [%2];"
                    : : "r"(s), "l"(g), "r"(data_addr)
                );
            }
            // Bulk loads complete through mbarrier, without a commit_group.
        }

        // ── CONSUMERS: wait for data, compute, then signal release ─────
        // All threads wait for data_bar (copy to complete)
        {
            uint32_t ready = 0;
            while (!ready) {
                asm volatile(
                    "{\n\t"
                    "  .reg .pred p;\n\t"
                    "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
                    "    p, [%1], %2;\n\t"
                    "  selp.u32 %0, 1, 0, p;\n\t"
                    "}\n\t"
                    : "=r"(ready) : "r"(data_addr), "r"((uint32_t)data_phase)
                );
            }
            data_phase ^= 1;
        }

        // Compute: multiply each element by 2.0 and write to dst
        for (int i = tid; i < TILE; i += nthreads) {
            float val = smem[i];
            dst[k * TILE + i] = val * 2.0f;
        }

        // All threads signal release_bar: "done reading this tile"
        // .release ensures the writes to dst are visible before the signal.
        asm volatile(
            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
            : : "r"(release_addr)
        );

        // Sync before next iteration so producer doesn't race ahead
        __syncthreads();
    }

    // Cleanup (optional but good practice)
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(data_addr));
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(release_addr));
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex02: WAR Prevention — Two-Barrier Pipeline             ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: requires Hopper (SM 9.0+)\n"); return 1;
    }

    const int N = K_TILES * TILE;
    float h_in[N], h_out[N], h_exp[N];
    for (int i = 0; i < N; i++) { h_in[i] = (float)i; h_exp[i] = i * 2.0f; }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(float)));

    war_prevention_kernel<<<1, TILE>>>(d_in, d_out, K_TILES);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - h_exp[i]) > 1e-4f) errors++;

    printf("Config: %d tiles × %d elements = %d total\n", K_TILES, TILE, N);
    printf("Element errors: %d / %d\n", errors, N);
    printf("Sample: in=%.0f → out=%.0f (expected %.0f)\n",
           h_in[7], h_out[7], h_exp[7]);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", errors == 0
           ? "✓ ALL PASS — WAR hazard correctly prevented!"
           : "✗ FAILURES — WAR hazard not correctly handled");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    return (errors == 0) ? 0 : 1;
}
