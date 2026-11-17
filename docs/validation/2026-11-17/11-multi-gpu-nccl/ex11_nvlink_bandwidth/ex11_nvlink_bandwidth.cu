// ═══════════════════════════════════════════════════════════════════
// ex11_nvlink_bandwidth.cu
//
// PURPOSE: Measure and compare GPU-to-GPU transfer bandwidth across
// three different code paths:
//   Path 1: Staged via CPU (pinned host buffer)
//   Path 2: cudaMemcpyPeer (DMA, uses NVLink if available)
//   Path 3: Kernel-based LD/ST (shows straw-vs-firehose effect)
//
// This demonstrates WHY raw P2P kernel access is insufficient and
// why NCCL uses DMA primitives to saturate NVLink bandwidth.
//
// WORKS ON: Any multi-GPU system. NVLink bonus visible on DGX/HGX.
//           Falls back gracefully if < 2 GPUs available.
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// ── Timing helper ─────────────────────────────────────────────────
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ── Kernel: LD/ST cross-GPU copy (reads src on remote GPU) ────────
// Each thread copies one float from remote (peer) GPU memory.
// This uses LOAD from GPU 0's perspective while src lives on GPU 1.
// With few blocks: shows "straw" — one NVLink lane.
// With many blocks: approaches NVLink saturation.
__global__ void peer_copy_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];   // load from peer GPU memory
    }
}

// ── Measure bandwidth for a given code path ───────────────────────
double measure_bandwidth_GBs(size_t bytes, float time_ms) {
    return (double)bytes / (time_ms * 1e-3) / 1e9;
}

// ── Print device info ─────────────────────────────────────────────
void print_device_info(int dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("  GPU %d: %s  (sm_%d%d, %d MB HBM, %d SMs)\n",
           dev, prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / 1024 / 1024),
           prop.multiProcessorCount);
}

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  ex11_nvlink_bandwidth — GPU-to-GPU Transfer Benchmark\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    printf("GPUs detected: %d\n", num_gpus);
    for (int i = 0; i < num_gpus; ++i) print_device_info(i);

    if (num_gpus < 2) {
        printf("\nNeed at least 2 GPUs for P2P benchmark. Exiting.\n");
        return 0;
    }

    const int dev0 = 0, dev1 = 1;
    const size_t N        = 32 * 1024 * 1024;   // 32M floats = 128 MB
    const size_t bytes    = N * sizeof(float);
    const int    WARMUP   = 3;
    const int    ITERS    = 10;

    printf("\nTransfer size: %.0f MB\n\n", bytes / 1e6);

    // ── Allocate GPU buffers ──────────────────────────────────────
    float *d0_src, *d0_dst;   // on GPU 0
    float *d1_src, *d1_dst;   // on GPU 1
    float *h_pinned;           // pinned host buffer for staging

    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMalloc(&d0_src, bytes));
    CHECK_CUDA(cudaMalloc(&d0_dst, bytes));
    CHECK_CUDA(cudaMemset(d0_src, 1, bytes));

    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMalloc(&d1_src, bytes));
    CHECK_CUDA(cudaMalloc(&d1_dst, bytes));
    CHECK_CUDA(cudaMemset(d1_src, 2, bytes));

    CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));   // pinned for DMA

    // ── Check and enable peer access ─────────────────────────────
    int can_peer = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_peer, dev0, dev1));
    printf("NVLink/Peer access available (dev%d ↔ dev%d): %s\n\n",
           dev0, dev1, can_peer ? "YES ✓" : "NO ✗ (will use PCIe fallback)");

    if (can_peer) {
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dev1, 0));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dev0, 0));
    }

    // ── Events for timing ─────────────────────────────────────────
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ══════════════════════════════════════════════════════════════
    // PATH 1: Staged via CPU (GPU0 → pinned host → GPU1)
    // ══════════════════════════════════════════════════════════════
    printf("─── Path 1: Staged via CPU (pinned host buffer) ────\n");

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaMemcpy(h_pinned, d0_src, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaMemcpy(d1_dst, h_pinned, bytes, cudaMemcpyHostToDevice));
    }

    float staged_ms = 0;
    for (int i = 0; i < ITERS; ++i) {
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_pinned, d0_src, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaMemcpy(d1_dst, h_pinned, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        staged_ms += elapsed_ms(start, stop);
    }
    staged_ms /= ITERS;
    double staged_GBs = measure_bandwidth_GBs(bytes, staged_ms);
    printf("  Round-trip time:  %.2f ms\n", staged_ms);
    printf("  Effective BW:     %.1f GB/s (%.0f MB/s)\n\n",
           staged_GBs, staged_GBs * 1e3);

    // ══════════════════════════════════════════════════════════════
    // PATH 2: cudaMemcpyPeer (NVLink DMA if available)
    // ══════════════════════════════════════════════════════════════
    printf("─── Path 2: cudaMemcpyPeer (NVLink DMA path) ───────\n");

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        CHECK_CUDA(cudaMemcpyPeer(d1_dst, dev1, d0_src, dev0, bytes));
    }

    float peer_ms = 0;
    CHECK_CUDA(cudaSetDevice(dev0));
    for (int i = 0; i < ITERS; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpyPeer(d1_dst, dev1, d0_src, dev0, bytes));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        peer_ms += elapsed_ms(start, stop);
    }
    peer_ms /= ITERS;
    double peer_GBs = measure_bandwidth_GBs(bytes, peer_ms);
    printf("  Transfer time:    %.2f ms\n", peer_ms);
    printf("  Effective BW:     %.1f GB/s  %s\n\n",
           peer_GBs,
           can_peer ? "(NVLink)" : "(PCIe fallback)");

    // ══════════════════════════════════════════════════════════════
    // PATH 3: Kernel-based LD/ST (straw vs firehose)
    // Kernel on GPU 0 reads from GPU 1's memory (peer mapped)
    // ══════════════════════════════════════════════════════════════
    printf("─── Path 3: Kernel LD/ST (straw vs firehose) ───────\n");

    if (!can_peer) {
        printf("  Skipped: peer access not available.\n\n");
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev0);
        int total_sms = prop.multiProcessorCount;

        // Test with 1 SM and with all SMs
        const int sm_configs[] = {1, total_sms};
        for (int c = 0; c < 2; ++c) {
            int num_blocks = sm_configs[c];
            int threads    = 256;
            // Each block handles N/num_blocks elements
            size_t elems_per_block = (N + num_blocks - 1) / num_blocks;

            // Warmup
            CHECK_CUDA(cudaSetDevice(dev0));
            for (int i = 0; i < WARMUP; ++i) {
                peer_copy_kernel<<<num_blocks, threads>>>(d1_src, d0_dst, (int)N);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            float kernel_ms = 0;
            for (int i = 0; i < ITERS; ++i) {
                CHECK_CUDA(cudaEventRecord(start));
                peer_copy_kernel<<<num_blocks, threads>>>(d1_src, d0_dst, (int)N);
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                kernel_ms += elapsed_ms(start, stop);
            }
            kernel_ms /= ITERS;
            double kernel_GBs = measure_bandwidth_GBs(bytes, kernel_ms);
            printf("  %3d SM(s), %5d blocks: %.2f ms → %.1f GB/s  %s\n",
                   num_blocks, num_blocks, kernel_ms, kernel_GBs,
                   num_blocks == 1 ? "← \"straw\" (1 NVLink lane)" : "← \"firehose\" (all lanes)");
        }
        printf("\n");
    }

    // ── Summary ───────────────────────────────────────────────────
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SUMMARY: GPU 0 → GPU 1 bandwidth comparison\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Staged via CPU:     %.1f GB/s\n", staged_GBs);
    printf("  cudaMemcpyPeer:     %.1f GB/s  (%.1f× vs staged)\n",
           peer_GBs, peer_GBs / staged_GBs);
    if (can_peer)
        printf("  Note: NVLink theoretical peak: ~450 GB/s per direction\n");
    else
        printf("  Note: No NVLink detected. Both paths use PCIe.\n");
    printf("═══════════════════════════════════════════════════════\n");

    // Cleanup
    cudaSetDevice(dev0);
    if (can_peer) cudaDeviceDisablePeerAccess(dev1);
    cudaFree(d0_src); cudaFree(d0_dst);
    cudaSetDevice(dev1);
    if (can_peer) cudaDeviceDisablePeerAccess(dev0);
    cudaFree(d1_src); cudaFree(d1_dst);
    cudaFreeHost(h_pinned);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
