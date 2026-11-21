// ex11_nvlink_bandwidth.cu — complete-buffer P2P audit benchmark.
// Peer capability is not proof of NVLink. Record nvidia-smi topo -m separately.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

static inline void check_cuda_status(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s:%d: %s\n", file, line, cudaGetErrorString(status));
        exit(1);
    }
}
#define CHECK_CUDA(call) check_cuda_status((call), __FILE__, __LINE__)

__global__ void peer_copy_kernel(const float* __restrict__ src,
                                 float* __restrict__ dst, size_t n) {
    // Every launch covers all n, including the one-block configuration.
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x)
        dst[i] = src[i];
}

double rate(size_t bytes, double ms) {
    if (!(ms > 0.0)) { fprintf(stderr, "Invalid elapsed time\n"); exit(1); }
    return (double)bytes / (ms * 1e6);
}

float elapsed(cudaEvent_t begin, cudaEvent_t end) {
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, begin, end));
    return ms;
}

// Copy back and inspect every byte outside the timed region.
bool verify(int owner, const void* dst, unsigned char* host,
            size_t bytes, unsigned char expected, const char* label) {
    CHECK_CUDA(cudaSetDevice(owner));
    CHECK_CUDA(cudaMemcpy(host, dst, bytes, cudaMemcpyDeviceToHost));
    size_t errors = 0;
    for (size_t i = 0; i < bytes; ++i) {
        if (host[i] != expected) {
            if (errors < 4) fprintf(stderr, "%s byte %zu: got %u expected %u\n",
                                   label, i, (unsigned)host[i], (unsigned)expected);
            ++errors;
        }
    }
    printf("  %s full-buffer verification: %s (%zu byte errors)\n",
           label, errors == 0 ? "PASS" : "FAIL", errors);
    return errors == 0;
}

int main() {
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    printf("GPUs detected: %d\n", count);
    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, dev));
        printf("GPU %d: %s, SM %d.%d, %d SMs\n",
               dev, p.name, p.major, p.minor, p.multiProcessorCount);
    }
    if (count < 2) { fprintf(stderr, "Requires at least 2 GPUs\n"); return 2; }
    const int dev0 = 0, dev1 = 1, warmup = 3, iters = 10;
    const size_t n = 32 * 1024 * 1024, bytes = n * sizeof(float);
    float *src0, *dst0, *src1, *dst1;
    unsigned char* host;
    CHECK_CUDA(cudaMallocHost(&host, bytes));
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMalloc(&src0, bytes)); CHECK_CUDA(cudaMalloc(&dst0, bytes));
    CHECK_CUDA(cudaMemset(src0, 0x01, bytes));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMalloc(&src1, bytes)); CHECK_CUDA(cudaMalloc(&dst1, bytes));
    CHECK_CUDA(cudaMemset(src1, 0x02, bytes));
    // Complete GPU 1 initialization before GPU 0 can read its peer source.
    CHECK_CUDA(cudaDeviceSynchronize());

    int can01 = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    printf("GPU 0 can access GPU 1 memory: %s (does not identify NVLink)\n",
           can01 ? "YES" : "NO");
    printf("Record topology with nvidia-smi topo -m. Payload: %zu bytes\n", bytes);
    bool enabled_by_us = false;
    CHECK_CUDA(cudaSetDevice(dev0));
    if (can01) {
        cudaError_t e = cudaDeviceEnablePeerAccess(dev1, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) {
            (void)cudaGetLastError(); // consume the expected already-enabled status
        } else {
            CHECK_CUDA(e);
            enabled_by_us = true;
        }
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    bool ok = true;

    // Staged path, GPU 0 -> host -> GPU 1; host timing covers both devices.
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemset(dst1, 0, bytes)); CHECK_CUDA(cudaDeviceSynchronize());
    double staged_ms = 0;
    for (int i = -warmup; i < iters; ++i) {
        auto begin = std::chrono::steady_clock::now();
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaMemcpy(host, src0, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaMemcpy(dst1, host, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();
        if (i >= 0) staged_ms += std::chrono::duration<double, std::milli>(end-begin).count();
    }
    staged_ms /= iters;
    ok &= verify(dev1, dst1, host, bytes, 0x01, "Staged 0->1");
    printf("  Staged complete-transfer time: %.6f ms, payload %.3f GB/s\n",
           staged_ms, rate(bytes, staged_ms));

    // Peer-copy API; the runtime chooses the path supported by this system.
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemset(dst1, 0, bytes)); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(dev0));
    double peer_ms = 0;
    for (int i = -warmup; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpyPeerAsync(dst1, dev1, src0, dev0, bytes, 0));
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        if (i >= 0) peer_ms += elapsed(start, stop);
    }
    peer_ms /= iters;
    ok &= verify(dev1, dst1, host, bytes, 0x01, "Peer API 0->1");
    printf("  Peer API transfer time: %.6f ms, payload %.3f GB/s (link unclassified)\n",
           peer_ms, rate(bytes, peer_ms));

    if (can01) {
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, dev0));
        const int grids[] = {1, p.multiProcessorCount};
        for (int blocks : grids) {
            CHECK_CUDA(cudaSetDevice(dev0));
            CHECK_CUDA(cudaMemset(dst0, 0, bytes)); CHECK_CUDA(cudaDeviceSynchronize());
            double ms = 0;
            for (int i = -warmup; i < iters; ++i) {
                CHECK_CUDA(cudaEventRecord(start));
                peer_copy_kernel<<<blocks, 256>>>(src1, dst0, n);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
                if (i >= 0) ms += elapsed(start, stop);
            }
            ms /= iters;
            ok &= verify(dev0, dst0, host, bytes, 0x02, "Kernel 1->0");
            printf("  Kernel grid %d blocks: %.6f ms, payload %.3f GB/s\n",
                   blocks, ms, rate(bytes, ms));
        }
    } else printf("Kernel peer reads skipped: GPU 0 cannot access GPU 1 memory.\n");

    // Events belong to GPU 0; destroy them while their owner is current.
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    if (enabled_by_us) CHECK_CUDA(cudaDeviceDisablePeerAccess(dev1));
    CHECK_CUDA(cudaFree(src0)); CHECK_CUDA(cudaFree(dst0));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaFree(src1)); CHECK_CUDA(cudaFree(dst1));
    CHECK_CUDA(cudaFreeHost(host));
    printf("Full-buffer checks: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
