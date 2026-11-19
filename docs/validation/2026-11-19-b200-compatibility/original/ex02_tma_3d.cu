// ═══════════════════════════════════════════════════════════════════
// ex02_tma_3d.cu
//
// PURPOSE: Demonstrate TMA with a 3D tensor (attention use case).
//   Batch × Sequence × HeadDim tensor, loading [1][SEQ_T][HD] tiles.
//
// Dimension ordering (CRITICAL — innermost first in TMA):
//   globalDim[0] = HEAD_DIM  (fastest varying — innermost)
//   globalDim[1] = SEQ_LEN   (middle dimension)
//   globalDim[2] = BATCH     (slowest varying — outermost)
//
// REQUIRES: sm_90a or sm_100a
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} } while(0)
#define CHECK_CU(call) do { CUresult _r=(call); if(_r!=CUDA_SUCCESS){ \
    const char* _s; cuGetErrorString(_r,&_s); \
    fprintf(stderr,"CU %s:%d %s\n",__FILE__,__LINE__,_s); exit(1);} } while(0)

static constexpr int BATCH    = 4;
static constexpr int SEQ_LEN  = 128;
static constexpr int HEAD_DIM = 64;
static constexpr int SEQ_TILE = 32;    // tile along sequence dim

static constexpr int TILE_ELEMS = HEAD_DIM * SEQ_TILE * 1;  // 64*32*1=2048
static constexpr int TILE_BYTES = TILE_ELEMS * sizeof(__half);  // 4096

__global__ void tma_3d_kernel(
    const __grid_constant__ CUtensorMap tma_desc,
    uint32_t* pass_count)
{
    __shared__ __align__(8)    uint64_t mbar;
    __shared__ __align__(1024) __half   smem_tile[SEQ_TILE][HEAD_DIM];

    int tid = threadIdx.x;
    uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);

    // blockIdx encodes (batch_idx, seq_tile_idx):
    int batch_idx    = blockIdx.y;
    int seq_tile_idx = blockIdx.x;
    int seq_start    = seq_tile_idx * SEQ_TILE;

    // Init barrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(mbar_addr));
    }
    // Publish initialized barrier state to the bulk-async proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // Producer: set expected tx and launch TMA
    if (tid == 0) {
        uint32_t smem_dest = __cvta_generic_to_shared(&smem_tile[0][0]);
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
            : : "r"(mbar_addr), "r"((uint32_t)TILE_BYTES)
        );
        // 3D TMA: coordinates = {hd_start=0, seq_start, batch_start}
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3, %4}], [%5];"
            :
            : "r"(smem_dest), "l"(&tma_desc),
              "r"(0),           // coord[0] = head_dim start
              "r"(seq_start),   // coord[1] = sequence start
              "r"(batch_idx),   // coord[2] = batch index
              "r"(mbar_addr)
            : "memory"
        );
    }

    // Wait for TMA
    {
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t .reg .pred p;\n\t"
                "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0;\n\t"
                "  selp.u32 %0, 1, 0, p;\n\t }\n\t"
                : "=r"(done) : "r"(mbar_addr)
            );
        }
    }

    // Verify: tensor[batch][seq][hd] = batch*SEQ_LEN*HEAD_DIM + seq*HEAD_DIM + hd
    int local_pass = 0;
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int s = i / HEAD_DIM;
        int h = i % HEAD_DIM;
        float expected = (float)(batch_idx * SEQ_LEN * HEAD_DIM
                               + (seq_start + s) * HEAD_DIM + h);
        // Match both the TMA swizzle and FP16 rounding in the source.
        expected = __half2float(__float2half(expected));
        int offset = i * sizeof(__half);
        offset ^= (offset >> 3) & 0x70;
        float got = __half2float(*reinterpret_cast<__half*>(reinterpret_cast<char*>(smem_tile) + offset));
        if (fabsf(got - expected) < 0.6f) local_pass++;
    }
    atomicAdd(pass_count, (uint32_t)local_pass);
    __syncthreads();

    if (tid == 0) {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(mbar_addr));
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex02: 3D TMA — Attention Tensor [B][S][D] Tile Load     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) { fprintf(stderr,"ERROR: needs SM 9.0+\n"); return 1; }

    CHECK_CU(cuInit(0));
    CUdevice cud; CUcontext cuc;
    CHECK_CU(cuDeviceGet(&cud, 0)); CHECK_CU(cuCtxCreate(&cuc, 0, cud));

    const int N = BATCH * SEQ_LEN * HEAD_DIM;
    __half* h_tensor = new __half[N];
    for (int i = 0; i < N; i++) h_tensor[i] = __float2half((float)i);

    __half* d_tensor;
    CHECK_CUDA(cudaMalloc(&d_tensor, N * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_tensor, h_tensor, N*sizeof(__half), cudaMemcpyHostToDevice));

    // ── Descriptor for [BATCH][SEQ_LEN][HEAD_DIM] FP16 tensor ────
    CUtensorMap tma_desc;
    // globalDim: innermost first → {HEAD_DIM, SEQ_LEN, BATCH}
    cuuint64_t gdim[3]    = {HEAD_DIM, SEQ_LEN, BATCH};
    // globalStrides: bytes to advance dim[1] (SEQ) and dim[2] (BATCH)
    cuuint64_t gstride[2] = {
        (cuuint64_t)HEAD_DIM * sizeof(__half),                      // bytes per seq step
        (cuuint64_t)SEQ_LEN * HEAD_DIM * sizeof(__half)             // bytes per batch step
    };
    cuuint32_t bdim[3]    = {HEAD_DIM, SEQ_TILE, 1};  // tile: all head_dim, SEQ_TILE seq, 1 batch
    cuuint32_t estride[3] = {1, 1, 1};

    printf("3D Tensor: [%d][%d][%d] FP16\n", BATCH, SEQ_LEN, HEAD_DIM);
    printf("Tile:      [1][%d][%d] = %d elements = %d bytes\n\n",
           SEQ_TILE, HEAD_DIM, TILE_ELEMS, TILE_BYTES);

    CHECK_CU(cuTensorMapEncodeTiled(
        &tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        d_tensor,
        gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    ));
    printf("Descriptor created successfully.\n\n");

    int seq_tiles = SEQ_LEN / SEQ_TILE;  // = 4
    dim3 grid(seq_tiles, BATCH);          // grid = (4 seq tiles, 4 batches) = 16 blocks
    int  block = 64;

    uint32_t *d_passes, h_passes = 0;
    CHECK_CUDA(cudaMalloc(&d_passes, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_passes, 0, sizeof(uint32_t)));

    printf("Launching %d blocks (%d seq_tiles × %d batches)...\n",
           seq_tiles * BATCH, seq_tiles, BATCH);

    tma_3d_kernel<<<grid, block>>>(tma_desc, d_passes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_passes, d_passes, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t expected = (uint32_t)BATCH * seq_tiles * TILE_ELEMS;

    printf("Element checks: %u / %u passed\n", h_passes, expected);
    bool ok = (h_passes == expected);
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", ok ? "✓ ALL PASS" : "✗ FAILURES");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    delete[] h_tensor;
    CHECK_CUDA(cudaFree(d_tensor)); CHECK_CUDA(cudaFree(d_passes));
    return ok ? 0 : 1;
}
