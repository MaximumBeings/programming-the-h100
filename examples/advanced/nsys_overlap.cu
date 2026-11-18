// Compare one stream with two nonblocking streams; validate every output.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
__global__ void transform(const float* in, float* out, int n) {
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = 2.0f * in[i] + 1.0f;
}
int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "serial") && std::strcmp(argv[1], "overlap"))) {
        std::fprintf(stderr, "Usage: nsys_overlap serial|overlap\n"); return 2;
    }
    try {
        constexpr int chunk = 1 << 18, chunks = 16, repetitions = 8;
        const int slots = std::strcmp(argv[1], "serial") == 0 ? 1 : 2;
        const size_t count = size_t(chunk) * chunks, bytes = size_t(chunk) * sizeof(float);
        float *input = nullptr, *output = nullptr, *di[2]{}, *dout[2]{};
        cudaStream_t streams[2]{};
        cudaEvent_t start{}, stop{}, done[2]{};
        check(cudaMallocHost(reinterpret_cast<void**>(&input), count * sizeof(float)));
        check(cudaMallocHost(reinterpret_cast<void**>(&output), count * sizeof(float)));
        for (size_t i = 0; i < count; ++i) input[i] = float(i % 1009);
        for (int s = 0; s < slots; ++s) {
            check(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&done[s], cudaEventDisableTiming));
            check(cudaMalloc(reinterpret_cast<void**>(&di[s]), bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&dout[s]), bytes));
        }
        // Warm the actual kernel before the captured, timed region.
        check(cudaMemcpy(di[0], input, bytes, cudaMemcpyHostToDevice));
        transform<<<(chunk + 255) / 256, 256, 0, streams[0]>>>(di[0], dout[0], chunk);
        check(cudaGetLastError()); check(cudaStreamSynchronize(streams[0]));
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        check(cudaProfilerStart());
        check(cudaEventRecord(start));
        for (int s = 0; s < slots; ++s) check(cudaStreamWaitEvent(streams[s], start));
        for (int rep = 0; rep < repetitions; ++rep)
            for (int c = 0; c < chunks; ++c) {
                int s = c % slots;
                check(cudaMemcpyAsync(di[s], input + size_t(c) * chunk, bytes, cudaMemcpyHostToDevice, streams[s]));
                transform<<<(chunk + 255) / 256, 256, 0, streams[s]>>>(di[s], dout[s], chunk);
                check(cudaGetLastError());
                check(cudaMemcpyAsync(output + size_t(c) * chunk, dout[s], bytes, cudaMemcpyDeviceToHost, streams[s]));
            }
        for (int s = 0; s < slots; ++s) {
            check(cudaEventRecord(done[s], streams[s]));
            check(cudaStreamWaitEvent(nullptr, done[s]));
        }
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        check(cudaProfilerStop());
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, start, stop));
        size_t errors = 0;
        for (size_t i = 0; i < count; ++i) errors += output[i] != 2.0f * input[i] + 1.0f;
        std::printf("%s: mode=%s slots=%d checked=%zu errors=%zu device_span_ms=%.3f\n",
                    errors ? "OVERLAP FAIL" : "OVERLAP PASS", argv[1], slots, count, errors, milliseconds);
        for (int s = 0; s < slots; ++s) {
            check(cudaFree(di[s])); check(cudaFree(dout[s]));
            check(cudaEventDestroy(done[s])); check(cudaStreamDestroy(streams[s]));
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        check(cudaFreeHost(input)); check(cudaFreeHost(output));
        return errors ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "OVERLAP ERROR: %s\n", e.what()); return 1;
    }
}
