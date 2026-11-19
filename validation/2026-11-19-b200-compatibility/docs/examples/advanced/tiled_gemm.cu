// A deliberately simple SIMT shared-memory GEMM, not Tensor Core or CuTe code.
// D = alpha*A*B + beta*C, row-major padded storage, independent batches.
// Small binary-rational inputs make these bounded reference cases exact in FP32.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void cuda_check(cudaError_t result, char const* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(result));
        throw std::runtime_error("CUDA operation failed");
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call)

constexpr int tile = 16;
__global__ void tiled_gemm(float const* a, float const* b, float const* c, float* d,
                           int m, int n, int k, int lda, int ldb, int ldc,
                           int stride_a, int stride_b, int stride_c,
                           float alpha, float beta) {
    __shared__ float a_tile[tile][tile];
    __shared__ float b_tile[tile][tile];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * tile + ty;
    int column = blockIdx.x * tile + tx;
    int batch = blockIdx.z;
    float sum = 0.0f;
    for (int base = 0; base < k; base += tile) {
        int a_k = base + tx, b_k = base + ty;
        a_tile[ty][tx] = row < m && a_k < k ?
            a[batch * stride_a + row * lda + a_k] : 0.0f;
        b_tile[ty][tx] = b_k < k && column < n ?
            b[batch * stride_b + b_k * ldb + column] : 0.0f;
        __syncthreads();  // All writers complete before any tile is consumed.
        for (int inner = 0; inner < tile; ++inner)
            sum += a_tile[ty][inner] * b_tile[inner][tx];
        __syncthreads();  // All readers complete before the next tile overwrites.
    }
    if (row < m && column < n) {
        int offset = batch * stride_c + row * ldc + column;
        d[offset] = alpha * sum + beta * c[offset];
    }
}

static int run_case(int m, int n, int k, int batches) {
    constexpr int guard = 16;
    constexpr float canary = -98765.0f;
    constexpr float alpha = 0.5f, beta = 0.25f;
    int lda = k + 3, ldb = n + 5, ldc = n + 7;
    int stride_a = m * lda, stride_b = k * ldb, stride_c = m * ldc;
    std::vector<float> a(guard + batches * stride_a + guard, canary);
    std::vector<float> b(guard + batches * stride_b + guard, canary);
    std::vector<float> c(guard + batches * stride_c + guard, canary);
    std::vector<float> d(c.size(), canary), expected(c.size(), canary);
    for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < m; ++row)
            for (int inner = 0; inner < k; ++inner)
                a[guard + batch * stride_a + row * lda + inner] =
                    float((row * 3 + inner * 5 + batch * 7) % 9 - 4) * 0.125f;
        for (int inner = 0; inner < k; ++inner)
            for (int column = 0; column < n; ++column)
                b[guard + batch * stride_b + inner * ldb + column] =
                    float((inner * 2 + column * 3 + batch * 5) % 7 - 3) * 0.25f;
        for (int row = 0; row < m; ++row)
            for (int column = 0; column < n; ++column) {
                int offset = guard + batch * stride_c + row * ldc + column;
                c[offset] = float((row + column * 2 + batch) % 5 - 2) * 0.5f;
                d[offset] = std::numeric_limits<float>::quiet_NaN();
            }
    }
    // Independent scalar CPU calculation: no shared tiling or GPU fragment map.
    // Products are multiples of 1/32 and bounded short sums remain FP32-exact.
    for (int batch = 0; batch < batches; ++batch)
        for (int row = 0; row < m; ++row)
            for (int column = 0; column < n; ++column) {
                double sum = 0.0;
                for (int inner = 0; inner < k; ++inner)
                    sum += double(a[guard + batch * stride_a + row * lda + inner]) *
                           double(b[guard + batch * stride_b + inner * ldb + column]);
                int offset = guard + batch * stride_c + row * ldc + column;
                expected[offset] = float(double(alpha) * sum + double(beta) * c[offset]);
            }
    float *da = nullptr, *db = nullptr, *dc = nullptr, *dd = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dd, d.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(tile, tile), grid((n + tile - 1) / tile, (m + tile - 1) / tile, batches);
    tiled_gemm<<<grid, block>>>(da + guard, db + guard, dc + guard, dd + guard,
        m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d.data(), dd, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> a_after(a.size()), b_after(b.size()), c_after(c.size());
    CUDA_CHECK(cudaMemcpy(a_after.data(), da, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_after.data(), db, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_after.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(da));
    int errors = 0;
    for (size_t i = 0; i < d.size(); ++i)
        if (!std::isfinite(d[i]) || d[i] != expected[i]) {
            if (errors < 4)
                std::fprintf(stderr, "D storage[%zu]=%.9g expected=%.9g\n", i, d[i], expected[i]);
            ++errors;
        }
    for (size_t i = 0; i < a.size(); ++i) if (a_after[i] != a[i]) ++errors;
    for (size_t i = 0; i < b.size(); ++i) if (b_after[i] != b[i]) ++errors;
    for (size_t i = 0; i < c.size(); ++i) if (c_after[i] != c[i]) ++errors;
    std::printf("SIMT GEMM M=%d N=%d K=%d batches=%d: %d exact outputs, padding/canaries/inputs checked, errors=%d\n",
                m, n, k, batches, m * n * batches, errors);
    return errors;
}

int main() {
    try {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (!count) { std::printf("BLOCKED: requires one CUDA GPU\n"); return 2; }
        CUDA_CHECK(cudaSetDevice(0));
        int errors = run_case(16, 16, 16, 1);
        errors += run_case(37, 29, 41, 2);
        errors += run_case(7, 9, 0, 1);
        std::printf("TILED GEMM %s: 3 cases, errors=%d\n", errors ? "FAIL" : "PASS", errors);
        return errors ? 1 : 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "TILED GEMM FAIL: %s\n", error.what());
        return 1;
    }
}
