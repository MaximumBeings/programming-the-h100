// ═══════════════════════════════════════════════════════════════════
// ex01_tma_basic.cu
//
// PURPOSE: Demonstrate cuTensorMap setup and TMA tile loading.
//
// WHAT THIS SHOWS:
//   1. cuTensorMapEncodeTiled — all 12 parameters explained
//   2. Kernel with TMA cp.async.bulk.tensor
//   3. mbarrier synchronization after TMA
//   4. Verification of loaded tile contents
//
// KERNEL: Each block loads one 64×32 FP32 tile using TMA,
//         then verifies the values match the source matrix.
//
// REQUIRES: sm_90a (H100) or sm_100a (Blackwell)
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda.h>              // for CUtensorMap and cuTensorMapEncodeTiled

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _s; cuGetErrorString(_r, &_s); \
        fprintf(stderr, "CU ERROR %s:%d: %s\n", \
                __FILE__, __LINE__, _s); exit(1); } } while(0)

// Tile dimensions
static constexpr int TILE_COLS = 32;    // inner dimension (columns)
static constexpr int TILE_ROWS = 32;    // outer dimension (rows)
static constexpr int TILE_ELEMS = TILE_COLS * TILE_ROWS;
static constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);

// Matrix dimensions
static constexpr int MAT_COLS = 256;   // inner dimension
static constexpr int MAT_ROWS = 128;   // outer dimension

// ─────────────────────────────────────────────────────────────────
// KERNEL: each block loads one tile using TMA, verifies it
// ─────────────────────────────────────────────────────────────────
__global__ void tma_load_kernel(
    const __grid_constant__ CUtensorMap tma_desc,  // descriptor by value
    uint32_t* __restrict__ pass_count,
    int num_tile_cols)  // number of tiles per row
{
    // ── Shared memory ─────────────────────────────────────────────
    // The mbarrier MUST be 8-byte aligned.
    // The SMEM tile is where TMA will write the loaded data.
    __shared__ __align__(8) uint64_t mbar;
    // Swizzle_128B requires SMEM to be 1024-byte aligned for correctness.
    // Using __align__(1024) ensures TMA writes land in the right banks.
    __shared__ __align__(1024) float smem_tile[TILE_ROWS][TILE_COLS];

    int tid = threadIdx.x;
    uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);

    // ── Determine this block's tile coordinates ───────────────────
    // blockIdx.x encodes the tile: blockIdx.x = tile_row * num_tile_cols + tile_col
    int tile_row = blockIdx.x / num_tile_cols;
    int tile_col = blockIdx.x % num_tile_cols;
    int col_start = tile_col * TILE_COLS;
    int row_start = tile_row * TILE_ROWS;

    // ── STEP 1: Initialize mbarrier ───────────────────────────────
    // Count = 1: only the elected producer thread calls arrive.expect_tx.
    if (tid == 0) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;"
            : : "r"(mbar_addr)
        );
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // ── STEP 2: Producer thread sets up TMA ───────────────────────
    if (tid == 0) {
        uint32_t smem_dest = __cvta_generic_to_shared(&smem_tile[0][0]);

        // arrive.expect_tx: signal arrival AND set expected byte count.
        // The barrier will complete when TMA delivers TILE_BYTES bytes.
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
            : : "r"(mbar_addr), "r"((uint32_t)TILE_BYTES)
        );

        // Issue the TMA load. Coordinates = top-left corner of tile.
        // Format: {innermost_coord, outermost_coord}
        // = {column_start, row_start} for a 2D matrix.
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(smem_dest),       // SMEM destination (32-bit shared addr)
              "l"(&tma_desc),       // descriptor address (64-bit)
              "r"(col_start),       // coordinate[0] = column (innermost)
              "r"(row_start),       // coordinate[1] = row (outermost)
              "r"(mbar_addr)        // mbarrier for completion tracking
            : "memory"
        );
    }

    // ── STEP 3: Wait for TMA to complete ─────────────────────────
    // All threads spin until the barrier fires (both thread count
    // AND byte count have reached zero).
    {
        uint32_t done = 0;
        int phase = 0;  // wait for phase 0 to complete (flip to 1)
        while (!done) {
            asm volatile(
                "{\n\t"
                "  .reg .pred p;\n\t"
                "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
                "    p, [%1], %2;\n\t"
                "  selp.u32 %0, 1, 0, p;\n\t"
                "}\n\t"
                : "=r"(done)
                : "r"(mbar_addr), "r"((uint32_t)phase)
            );
        }
    }

    // After try_wait.acquire returns, all TMA writes to smem_tile are
    // visible to this thread. Safe to read smem_tile now.

    // ── STEP 4: Verify tile contents ─────────────────────────────
    // Expected: smem_tile[r][c] = (float)((row_start + r) * MAT_COLS + (col_start + c))
    // This is how we initialized the source matrix on the host.
    int local_pass = 0;
    for (int i = tid; i < TILE_ELEMS; i += blockDim.x) {
        int r = i / TILE_COLS;
        int c = i % TILE_COLS;
        float expected = (float)((row_start + r) * MAT_COLS + (col_start + c));
        // Undo the 128B TMA swizzle before reading the logical element.
        int offset = i * sizeof(float);
        offset ^= (offset >> 3) & 0x70;
        float got = *reinterpret_cast<float*>(reinterpret_cast<char*>(smem_tile) + offset);
        if (fabsf(got - expected) < 0.5f) local_pass++;
    }
    atomicAdd(pass_count, (uint32_t)local_pass);

    // All waiters must finish using the barrier before invalidation.
    __syncthreads();
    // Cleanup
    if (tid == 0) {
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(mbar_addr));
    }
}

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ex01: cuTensorMap Setup + TMA Load + Verify             ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Check hardware
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: TMA requires Hopper (SM 9.0+). "
                        "Your GPU is SM %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    // Initialize CUDA driver API (needed for cuTensorMapEncodeTiled)
    CHECK_CU(cuInit(0));
    CUdevice cu_device; CUcontext cu_context;
    CHECK_CU(cuDeviceGet(&cu_device, 0));
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    const int N_ELEMS = MAT_COLS * MAT_ROWS;
    const int N_BYTES = N_ELEMS * sizeof(float);

    // Create and initialize host matrix: matrix[r][c] = r * MAT_COLS + c
    float* h_matrix = new float[N_ELEMS];
    for (int i = 0; i < N_ELEMS; i++) h_matrix[i] = (float)i;

    // Allocate device matrix (cudaMalloc ≥ 256B aligned → satisfies 16B req)
    float* d_matrix;
    CHECK_CUDA(cudaMalloc(&d_matrix, N_BYTES));
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, N_BYTES, cudaMemcpyHostToDevice));

    // ── Create the cuTensorMap descriptor ─────────────────────────
    // This is the central object of this lesson.
    CUtensorMap tma_desc;

    // [5] globalDim: {cols, rows} — innermost first!
    cuuint64_t global_dim[2]    = {MAT_COLS, MAT_ROWS};
    // [6] globalStrides: bytes per row (rank-1 entries)
    cuuint64_t global_stride[1] = {(cuuint64_t)MAT_COLS * sizeof(float)};
    // [7] boxDim: {tile_cols, tile_rows}
    cuuint32_t box_dim[2]       = {TILE_COLS, TILE_ROWS};
    // [8] elementStrides: dense (all 1s)
    cuuint32_t elem_stride[2]   = {1, 1};

    printf("Descriptor setup:\n");
    printf("  Matrix:       %d × %d FP32\n", MAT_COLS, MAT_ROWS);
    printf("  Tile:         %d × %d = %d elements = %d bytes\n",
           TILE_COLS, TILE_ROWS, TILE_ELEMS, TILE_BYTES);
    printf("  globalDim:    {%llu, %llu}\n", global_dim[0], global_dim[1]);
    printf("  globalStride: {%llu} bytes\n", global_stride[0]);
    printf("  boxDim:       {%u, %u}\n", box_dim[0], box_dim[1]);
    printf("  swizzle:      128B\n");
    printf("  l2Promotion:  128B\n\n");

    CHECK_CU(cuTensorMapEncodeTiled(
        &tma_desc,                                    // [1] output
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,              // [2] element type
        2,                                            // [3] rank
        d_matrix,                                     // [4] base pointer
        global_dim,                                   // [5] shape
        global_stride,                                // [6] byte strides
        box_dim,                                      // [7] tile shape
        elem_stride,                                  // [8] element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,                // [9]  no interleave
        CU_TENSOR_MAP_SWIZZLE_128B,                   // [10] 128B swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,           // [11] 128B prefetch
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA // [12] NaN fill
    ));
    printf("cuTensorMapEncodeTiled: SUCCESS\n\n");

    // Launch: one block per tile
    int num_tile_cols = MAT_COLS / TILE_COLS;   // = 4
    int num_tile_rows = MAT_ROWS / TILE_ROWS;   // = 4
    int num_tiles = num_tile_cols * num_tile_rows; // = 16
    int block_size = 64;

    uint32_t *d_passes, h_passes = 0;
    CHECK_CUDA(cudaMalloc(&d_passes, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_passes, 0, sizeof(uint32_t)));

    printf("Launching %d tile blocks (%d×%d tiles)...\n",
           num_tiles, num_tile_cols, num_tile_rows);

    tma_load_kernel<<<num_tiles, block_size>>>(
        tma_desc, d_passes, num_tile_cols
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_passes, d_passes, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    uint32_t expected = (uint32_t)num_tiles * TILE_ELEMS;
    printf("Element checks: %u / %u passed\n", h_passes, expected);

    bool all_pass = (h_passes == expected);
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", all_pass
           ? "✓ ALL PASS — TMA loaded all tiles correctly!"
           : "✗ FAILURES — check descriptor parameters");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    delete[] h_matrix;
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_passes));
    return all_pass ? 0 : 1;
}
