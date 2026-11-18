/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Adapted predicate/TiledCopy construction from CUTLASS v4.3.6
// examples/cute/tutorial/tiled_copy_if.cu. This teaching lab uses scalar
// float copies and instrumented ownership counts, not vectorization or MMA.
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace cute;

static void cuda_check(cudaError_t result, char const* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(result));
        throw std::runtime_error("CUDA operation failed");
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call)

template<class Source, class Destination, class TiledCopy>
__global__ void copy_tile(Source source, Destination destination,
                          TiledCopy tiled_copy, int* writers, int columns) {
    auto logical_shape = shape(source);
    auto coordinates = make_identity_tensor(logical_shape);
    auto predicates = cute::lazy::transform(coordinates, [&](auto coordinate) {
        return elem_less(coordinate, logical_shape);
    });
    auto tile_shape = make_shape(Int<4>{}, Int<32>{});
    auto tile_coordinate = make_coord(blockIdx.x, blockIdx.y);
    auto source_tile = local_tile(source, tile_shape, tile_coordinate);
    auto destination_tile = local_tile(destination, tile_shape, tile_coordinate);
    auto predicate_tile = local_tile(predicates, tile_shape, tile_coordinate);
    auto coordinate_tile = local_tile(coordinates, tile_shape, tile_coordinate);
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thread_source = thread_copy.partition_S(source_tile);
    auto thread_destination = thread_copy.partition_D(destination_tile);
    auto thread_predicate = thread_copy.partition_S(predicate_tile);
    auto thread_coordinate = thread_copy.partition_S(coordinate_tile);
    copy_if(tiled_copy, thread_predicate, thread_source, thread_destination);
    // Instrumentation proves each valid destination has exactly one owner.
    // The same predicate protects both the copy and the ownership counter.
    for (int i = 0; i < size(thread_predicate); ++i) {
        if (thread_predicate(i)) {
            auto coordinate = thread_coordinate(i);
            atomicAdd(writers + int(get<0>(coordinate)) * columns +
                                  int(get<1>(coordinate)), 1);
        }
    }
}

static int run_case(int rows, int columns) {
    constexpr int guard = 16;
    constexpr float source_canary = -77777.0f, destination_canary = -54321.0f;
    int source_pitch = columns + 3, destination_pitch = columns + 5;
    std::vector<float> source(guard + rows * source_pitch + guard, source_canary);
    std::vector<float> destination(guard + rows * destination_pitch + guard,
                                   destination_canary);
    auto expected = destination;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            float value = float(row * columns + column + 1);
            source[guard + row * source_pitch + column] = value;
            destination[guard + row * destination_pitch + column] =
                std::numeric_limits<float>::quiet_NaN();
            expected[guard + row * destination_pitch + column] = value;
        }
    }
    float *device_source = nullptr, *device_destination = nullptr;
    int* device_writers = nullptr;
    CUDA_CHECK(cudaMalloc(&device_source, source.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_destination, destination.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_writers, rows * columns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_source, source.data(), source.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_destination, destination.data(),
                          destination.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_writers, 0, rows * columns * sizeof(int)));
    auto tensor_shape = make_shape(rows, columns);
    auto source_tensor = make_tensor(make_gmem_ptr(device_source + guard),
        make_layout(tensor_shape, make_stride(source_pitch, Int<1>{})));
    auto destination_tensor = make_tensor(make_gmem_ptr(device_destination + guard),
        make_layout(tensor_shape, make_stride(destination_pitch, Int<1>{})));
    auto threads = make_layout(make_shape(Int<4>{}, Int<32>{}),
                               make_stride(Int<32>{}, Int<1>{}));
    auto values = make_layout(make_shape(Int<1>{}, Int<1>{}));
    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                                     threads, values);
    dim3 grid((rows + 3) / 4, (columns + 31) / 32);
    copy_tile<<<grid, 128>>>(source_tensor, destination_tensor, tiled_copy,
                            device_writers, columns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(destination.data(), device_destination,
                          destination.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> source_after(source.size());
    std::vector<int> writers(rows * columns);
    CUDA_CHECK(cudaMemcpy(source_after.data(), device_source,
                          source.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(writers.data(), device_writers,
                          writers.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_writers));
    CUDA_CHECK(cudaFree(device_destination));
    CUDA_CHECK(cudaFree(device_source));
    int errors = 0;
    for (size_t i = 0; i < expected.size(); ++i)
        if (destination[i] != expected[i]) ++errors;
    for (size_t i = 0; i < source.size(); ++i)
        if (source_after[i] != source[i]) ++errors;
    for (int count : writers) if (count != 1) ++errors;
    std::printf("copy M=%d N=%d: %d unique values, padding/canaries/source checked, errors=%d\n",
                rows, columns, rows * columns, errors);
    return errors;
}

int main() {
    try {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (!count) { std::printf("BLOCKED: requires one CUDA GPU\n"); return 2; }
        CUDA_CHECK(cudaSetDevice(0));
        int errors = run_case(8, 64);
        errors += run_case(37, 70);
        errors += run_case(1, 1);
        std::printf("COPY %s: 3 cases, errors=%d\n", errors ? "FAIL" : "PASS", errors);
        return errors ? 1 : 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "COPY FAIL: %s\n", error.what());
        return 1;
    }
}
