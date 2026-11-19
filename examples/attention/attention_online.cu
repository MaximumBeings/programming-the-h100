// Educational FP32 online attention: one CUDA thread per query row.
// No CUTLASS, Tensor Cores, dropout, bias or performance claim.
// CPU-only check: clang++ -x c++ -std=c++17 -O2 \
//   -DATTENTION_HOST_ONLY attention_online.cu -o attention_host
// Hopper: nvcc -std=c++17 -O2 -arch=sm_90a attention_online.cu -o attention_online
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef ATTENTION_HOST_ONLY
#include <cuda_runtime.h>
#define ATTENTION_HD __host__ __device__
#else
#define ATTENTION_HD
#endif

constexpr int MAX_DV = 16;

struct Shape {
    int nq, nk, d, dv;
    bool causal;
    int query_offset;
};

// Q/K/V/O are packed row-major float arrays. Each row owns its z/l/m state.
ATTENTION_HD void online_row(const float* q,
                             const float* k,
                             const float* v,
                             float* out,
                             float* lse,
                             Shape s,
                             int i) {
    float m = -INFINITY, l = 0.0f;
    float z[MAX_DV] = {};
    const float scale = 1.0f / sqrtf(static_cast<float>(s.d));
    for (int j = 0; j < s.nk; ++j) {
        if (s.causal && j > i + s.query_offset)
            continue;
        float score = 0.0f;
        for (int c = 0; c < s.d; ++c)
            score += q[i * s.d + c] * k[j * s.d + c];
        score *= scale;
        const float next_m = fmaxf(m, score);
        const float alpha = l == 0.0f ? 0.0f : expf(m - next_m);
        const float weight = expf(score - next_m);
        l = alpha * l + weight;
        for (int c = 0; c < s.dv; ++c)
            z[c] = alpha * z[c] + weight * v[j * s.dv + c];
        m = next_m;
    }
    for (int c = 0; c < s.dv; ++c)
        out[i * s.dv + c] = l > 0.0f ? z[c] / l : NAN;
    lse[i] = l > 0.0f ? m + logf(l) : NAN;
}

#ifndef ATTENTION_HOST_ONLY
__global__ void attention_kernel(const float* q,
                                 const float* k,
                                 const float* v,
                                 float* out,
                                 float* lse,
                                 Shape s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s.nq)
        online_row(q, k, v, out, lse, s, i);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceBuffer {
    float* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data),
                              count * sizeof(float)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data)
            cudaFree(data);
    }

    void release() {
        float* allocation = data;
        data = nullptr;
        check_cuda(cudaFree(allocation));
    }
};
#endif

void validate(const Shape& s,
              const std::vector<float>& q,
              const std::vector<float>& k,
              const std::vector<float>& v) {
    if (s.nq < 1 || s.nq > 1024 || s.nk < 1 || s.nk > 1024 ||
        s.d < 1 || s.d > 64 || s.dv < 1 || s.dv > MAX_DV ||
        s.query_offset < 0 || s.query_offset > 1024 ||
        q.size() != size_t(s.nq * s.d) ||
        k.size() != size_t(s.nk * s.d) ||
        v.size() != size_t(s.nk * s.dv))
        throw std::invalid_argument(
            "Unsupported shape, offset or storage extent");
    for (const auto* input : {&q, &k, &v})
        for (float x : *input)
            if (!std::isfinite(x) || std::fabs(x) > 8.0f)
                throw std::invalid_argument(
                    "Inputs must be finite with |x| <= 8");
}

// Independent materialized binary64 reference from the ACTUAL float inputs.
void reference(const Shape& s,
               const std::vector<float>& q,
               const std::vector<float>& k,
               const std::vector<float>& v,
               std::vector<double>& out,
               std::vector<double>& lse) {
    for (int i = 0; i < s.nq; ++i) {
        std::vector<double> scores(s.nk, -INFINITY);
        double m = -INFINITY;
        for (int j = 0; j < s.nk; ++j) {
            if (s.causal && j > i + s.query_offset)
                continue;
            double dot = 0;
            for (int c = 0; c < s.d; ++c)
                dot += double(q[i * s.d + c]) * k[j * s.d + c];
            scores[j] = dot / std::sqrt(double(s.d));
            m = std::max(m, scores[j]);
        }
        double sum = 0;
        for (double& score : scores) {
            score = std::exp(score - m);
            sum += score;
        }
        for (int c = 0; c < s.dv; ++c) {
            double value = 0;
            for (int j = 0; j < s.nk; ++j)
                value += (scores[j] / sum) * v[j * s.dv + c];
            out[i * s.dv + c] = value;
        }
        lse[i] = m + std::log(sum);
    }
}

void compare(const std::vector<float>& actual,
             const std::vector<double>& expected,
             const char* label) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double tolerance =
            1e-4 + 1e-4 * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) ||
            !std::isfinite(expected[i]) ||
            std::fabs(double(actual[i]) - expected[i]) > tolerance)
            throw std::runtime_error(std::string(label) +
                                     " mismatch at " +
                                     std::to_string(i));
    }
}

void run_case(const char* name,
              Shape s,
              std::vector<float> q,
              std::vector<float> k,
              std::vector<float> v) {
    validate(s, q, k, v);
    std::vector<float> out(s.nq * s.dv, NAN), lse(s.nq, NAN);
    std::vector<double> expected(out.size()),
        expected_lse(lse.size());
    reference(s, q, k, v, expected, expected_lse);
#ifdef ATTENTION_HOST_ONLY
    // Executes the shared FP32 row function on the CPU; not a CUDA run.
    for (int i = 0; i < s.nq; ++i)
        online_row(q.data(),
                   k.data(),
                   v.data(),
                   out.data(),
                   lse.data(),
                   s,
                   i);
#else
    DeviceBuffer dq(q.size()), dk(k.size()), dv(v.size());
    DeviceBuffer dout(out.size()), dlse(lse.size());
    check_cuda(cudaMemcpy(dq.data,
                          q.data(),
                          q.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dk.data,
                          k.data(),
                          k.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dv.data,
                          v.data(),
                          v.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(
        cudaMemset(dout.data, 0xff, out.size() * sizeof(float)));
    check_cuda(
        cudaMemset(dlse.data, 0xff, lse.size() * sizeof(float)));
    attention_kernel<<<(s.nq + 127) / 128, 128>>>(
        dq.data, dk.data, dv.data, dout.data, dlse.data, s);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    check_cuda(cudaMemcpy(out.data(),
                          dout.data,
                          out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(lse.data(),
                          dlse.data,
                          lse.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    dq.release();
    dk.release();
    dv.release();
    dout.release();
    dlse.release();
#endif
    compare(out, expected, "O");
    compare(lse, expected_lse, "LSE");
    if (std::string(name) == "worked-full" ||
        std::string(name) == "worked-reordered") {
        compare(out, {1.6, 2.0, 1.4, 2.4}, "Closed-form O");
        compare(
            lse, {std::log(5.0), std::log(5.0)}, "Closed-form LSE");
    } else if (std::string(name) == "worked-causal") {
        compare(
            out, {1.0, 0.0, 1.0 / 3.0, 4.0 / 3.0}, "Closed-form O");
        compare(
            lse, {std::log(2.0), std::log(3.0)}, "Closed-form LSE");
    } else if (std::string(name) == "worked-offset-1") {
        compare(
            out, {2.0 / 3.0, 2.0 / 3.0, 1.4, 2.4}, "Closed-form O");
        compare(
            lse, {std::log(3.0), std::log(5.0)}, "Closed-form LSE");
    }
    std::printf(
        "PASS %s: all %zu O values and %zu LSE values checked; first O=[",
        name,
        out.size(),
        lse.size());
    for (int c = 0; c < std::min(s.dv, 2); ++c)
        std::printf("%s%.7g", c ? ", " : "", out[c]);
    std::puts("]");
}

int main() {
    try {
#ifndef ATTENTION_HOST_ONLY
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            std::fprintf(stderr,
                         "BLOCKED: CUDA device unavailable (%s)\n",
                         cudaGetErrorString(status));
            return 2;
        }
        check_cuda(cudaSetDevice(0));
#endif
        const float a = std::sqrt(2.0f) * std::log(2.0f);
        const std::vector<float> q = {a, 0, 0, a};
        const std::vector<float> k = {1, 0, 0, 1, 1, 1};
        const std::vector<float> v = {1, 0, 0, 2, 3, 4};
        run_case("worked-full", {2, 3, 2, 2, false, 0}, q, k, v);
        run_case("worked-causal", {2, 3, 2, 2, true, 0}, q, k, v);
        run_case("worked-offset-1", {2, 3, 2, 2, true, 1}, q, k, v);
        run_case("worked-reordered",
                 {2, 3, 2, 2, false, 0},
                 q,
                 {0, 1, 1, 0, 1, 1},
                 {0, 2, 1, 0, 3, 4});
        Shape s{17, 23, 8, 5, false, 0};
        std::vector<float> q2(s.nq * s.d), k2(s.nk * s.d),
            v2(s.nk * s.dv);
        for (size_t i = 0; i < q2.size(); ++i)
            q2[i] = (int(i * 7 % 19) - 9) * 0.03125f;
        for (size_t i = 0; i < k2.size(); ++i)
            k2[i] = (int(i * 3 % 17) - 8) * 0.0625f;
        for (size_t i = 0; i < v2.size(); ++i)
            v2[i] = (int(i * 5 % 23) - 11) * 0.0625f;
        run_case("rectangular-full", s, q2, k2, v2);
        s.causal = true;
        run_case("rectangular-causal", s, q2, k2, v2);
        s.query_offset = 1;
        run_case("rectangular-offset-1", s, q2, k2, v2);
#ifdef ATTENTION_HOST_ONLY
        std::puts(
            "HOST ROW CHECK PASS: shared FP32 arithmetic versus binary64 reference; CUDA compilation/execution not performed.");
#else
        std::puts(
            "CUDA ATTENTION PASS: seven cases; complete O/LSE checks; no performance claim.");
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
