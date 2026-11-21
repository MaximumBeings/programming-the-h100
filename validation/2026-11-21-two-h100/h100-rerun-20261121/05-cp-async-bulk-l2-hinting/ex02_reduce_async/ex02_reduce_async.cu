// ═══════════════════════════════════════════════════════════════════
// ex02_reduce_async.cu
//
// PURPOSE: Demonstrate cp.reduce.async for hardware-accelerated
//   atomic tile reduction. Each block adds its SMEM tile to a
//   global accumulator using TMA — no thread-driven atomics needed.
//
// REQUIRES: sm_90a (H100) or sm_100a (Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} } while(0)

static constexpr int TILE_ELEMS = 256;
static constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);
static constexpr int NUM_BLOCKS = 4;

__global__ void reduce_async_kernel(float* global_acc) {
    __shared__ __align__(16) float smem_partial[TILE_ELEMS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Each block fills SMEM with its block-specific value
    // Block 0: fills with 1.0, Block 1: with 2.0, etc.
    float my_val = (float)(bid + 1);
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x)
        smem_partial[i] = my_val;
    __syncthreads();

    // Publish every ordinary shared-memory store to the async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // Producer thread issues the async reduction
    if (tid == 0) {
        uint32_t smem_src = __cvta_generic_to_shared(smem_partial);
        uint64_t dst_addr = reinterpret_cast<uint64_t>(global_acc);

        // Create evict_first policy: output goes to L2 then evicted
        uint64_t write_policy;
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
            : "=l"(write_policy));

        // cp.reduce.async: atomically add this block's tile to global_acc
        // TMA reads smem_partial and does:
        //   for i in [0, TILE_ELEMS): global_acc[i] += smem_partial[i]
        // All operations are atomic, all done by TMA hardware.
        asm volatile(
            "cp.reduce.async.bulk.global.shared::cta"
            ".bulk_group.add.f32.L2::cache_hint"
            " [%0], [%1], %2, %3;"
            :
            : "l"(dst_addr),            // global accumulator (64-bit)
              "r"(smem_src),            // SMEM source (32-bit)
              "r"((uint32_t)TILE_BYTES),// bytes to reduce
              "l"(write_policy)         // L2 evict_first for output
            : "memory"
        );

        // Commit the reduction to a bulk_group
        asm volatile("cp.async.bulk.commit_group;");
    }

    // Wait for this block's reduction to complete
    // (in a real pipeline, you'd do other work here instead)
    asm volatile("cp.async.bulk.wait_group 0;");
    // After wait_group 0: all committed bulk_group operations done.
    // global_acc has been atomically updated with this block's contribution.
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex02: cp.reduce.async — Atomic Tile Reduction           ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) { fprintf(stderr,"Needs SM 9.0+\n"); return 1; }

    float *d_acc, *h_acc;
    h_acc = new float[TILE_ELEMS];
    CHECK_CUDA(cudaMalloc(&d_acc, TILE_BYTES));
    CHECK_CUDA(cudaMemset(d_acc, 0, TILE_BYTES));  // initialize to 0

    printf("Config: %d blocks each adding their value (1,2,3,4) to %d elements\n",
           NUM_BLOCKS, TILE_ELEMS);
    printf("Expected result: every element = 1+2+3+4 = 10.0\n\n");

    reduce_async_kernel<<<NUM_BLOCKS, 64>>>(d_acc);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_acc, d_acc, TILE_BYTES, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < TILE_ELEMS; i++)
        if (fabsf(h_acc[i] - 10.0f) > 1e-4f) errors++;

    printf("Errors: %d / %d elements\n", errors, TILE_ELEMS);
    printf("Sample: acc[0]=%.1f acc[127]=%.1f acc[255]=%.1f\n",
           h_acc[0], h_acc[127], h_acc[255]);

    bool ok = (errors == 0);
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", ok
           ? "✓ ALL PASS — cp.reduce.async works correctly!"
           : "✗ FAILURES — check reduction logic");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    delete[] h_acc;
    CHECK_CUDA(cudaFree(d_acc));
    return ok ? 0 : 1;
}
