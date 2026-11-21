// ═══════════════════════════════════════════════════════════════════
// ex12_nccl_allreduce.cu
//
// PURPOSE: End-to-end NCCL AllReduce demonstration.
//
// DESIGN: Single-process, multi-threaded approach.
//   One CPU thread per GPU. Each thread:
//     1. Selects its GPU
//     2. Participates in ncclCommInitRank (all threads together)
//     3. Fills gradient buffer with (rank+1) × ones
//     4. Calls ncclAllReduce
//     5. Verifies result = sum(1..N_GPUS) on all elements
//
// WHY THREADS (not MPI)?
//   In production, each GPU is a SEPARATE PROCESS (via MPI or SLURM).
//   For a single-node test harness, threads are simpler to compile
//   and run without MPI installed. The NCCL API is identical.
//
// WHAT TO OBSERVE:
//   - All GPUs produce the same correct result
//   - Record bandwidth for this message size and selected algorithm.
//   - A fixed-message ring AllReduce does not have inverse-N runtime;
//     its per-rank send volume is 2*(N-1)/N times the payload.
//
// WORKS ON: sm_80+. Requires NCCL >= 2.15.
// ═══════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR [rank %d] at %s:%d — %s\n", \
                g_rank_id, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(call) do { \
    ncclResult_t _err = (call); \
    if (_err != ncclSuccess) { \
        fprintf(stderr, "NCCL ERROR [rank %d] at %s:%d — %s\n", \
                g_rank_id, __FILE__, __LINE__, ncclGetErrorString(_err)); \
        exit(1); \
    } \
} while(0)

// Suppress rank-id warning in macros before thread init
static __thread int g_rank_id = -1;

// ── Shared state across all threads ──────────────────────────────
struct SharedState {
    int          n_gpus;
    int*         rank_ok;     // one slot per rank; read by main after pthread_join
    ncclUniqueId nccl_id;
    ncclComm_t*  comms;       // one per GPU
    pthread_barrier_t barrier; // synchronize all threads
};

// ── Per-thread arguments ──────────────────────────────────────────
struct ThreadArgs {
    int           rank;
    int           n_gpus;
    int           n_elems;
    SharedState*  shared;
};

// ── Worker thread: one per GPU ────────────────────────────────────
void* worker(void* arg) {
    ThreadArgs* a      = (ThreadArgs*)arg;
    int rank           = a->rank;
    int n_gpus         = a->n_gpus;
    int n_elems        = a->n_elems;
    SharedState* shared = a->shared;
    g_rank_id          = rank;  // for error macros

    // ── Step 1: Select GPU ────────────────────────────────────────
    CHECK_CUDA(cudaSetDevice(rank));

    // ── Step 2: Rank 0 generates ncclUniqueId, all ranks sync ────
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&shared->nccl_id));
    }
    // Barrier: wait for rank 0 to generate the ID
    pthread_barrier_wait(&shared->barrier);

    // ── Step 3: All ranks initialize communicator ─────────────────
    // In production, the ncclUniqueId is distributed via PMIx or MPI_Bcast.
    // Here, all threads share it via shared memory (single process).
    CHECK_NCCL(ncclCommInitRank(&shared->comms[rank], n_gpus,
                                shared->nccl_id, rank));

    // ── Step 4: Allocate GPU buffers ──────────────────────────────
    size_t bytes = n_elems * sizeof(float);
    float *d_buf, *d_result;
    CHECK_CUDA(cudaMalloc(&d_buf,    bytes));
    CHECK_CUDA(cudaMalloc(&d_result, bytes));

    // ── Step 5: Initialize gradient buffer: d_buf[i] = rank + 1 ──
    // Using cudaMemset-based trick: fill via kernel
    // (We write a simple fill kernel inline)
    {
        float fill_val = (float)(rank + 1);
        // Simple: use cudaMemset + host init + memcpy
        float* h_buf = (float*)malloc(bytes);
        for (int i = 0; i < n_elems; ++i) h_buf[i] = fill_val;
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
        free(h_buf);
    }

    // ── Step 6: Barrier before collective ─────────────────────────
    pthread_barrier_wait(&shared->barrier);

    // ── Step 7: Timing setup ──────────────────────────────────────
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int w = 0; w < 3; ++w) {
        CHECK_NCCL(ncclAllReduce(d_buf, d_result, n_elems,
                                 ncclFloat, ncclSum,
                                 shared->comms[rank], stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // ── Step 8: Timed AllReduce ───────────────────────────────────
    const int ITERS = 10;
    // Refill buf after warmup (warmup overwrites result buffer)
    {
        float fill_val = (float)(rank + 1);
        float* h_buf = (float*)malloc(bytes);
        for (int i = 0; i < n_elems; ++i) h_buf[i] = fill_val;
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
        free(h_buf);
    }

    float total_ms = 0;
    for (int i = 0; i < ITERS; ++i) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        CHECK_NCCL(ncclAllReduce(d_buf, d_result, n_elems,
                                 ncclFloat, ncclSum,
                                 shared->comms[rank], stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    float avg_ms = total_ms / ITERS;

    // ── Step 9: Verify result ─────────────────────────────────────
    float expected = (float)(n_gpus * (n_gpus + 1) / 2);  // sum 1..N
    float* h_result = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n_elems; ++i) {
        if (!std::isfinite(h_result[i]) || fabsf(h_result[i] - expected) > 0.01f) {
            ok = false;
            if (rank == 0) {
                printf("  ERROR: result[%d] = %.1f, expected %.1f\n",
                       i, h_result[i], expected);
            }
            break;
        }
    }

    shared->rank_ok[rank] = ok ? 1 : 0;

    // ── Step 10: Report ───────────────────────────────────────────
    // AllReduce bus bandwidth = 2×(N-1)/N × data_size / time
    // (accounts for the ring reduce-scatter + all-gather pattern)
    double bus_bw_GBs = 2.0 * (n_gpus - 1) / n_gpus * bytes / (avg_ms * 1e-3) / 1e9;
    double algbw_GBs  = (double)bytes / (avg_ms * 1e-3) / 1e9;

    pthread_barrier_wait(&shared->barrier);  // synchronize before printing
    if (rank == 0) {
        printf("\n  AllReduce results:\n");
        printf("    N_GPUs:     %d\n", n_gpus);
        printf("    Data size:  %.2f MB (%.0fM elements × float)\n",
               bytes / 1e6, (double)n_elems / 1e6);
        printf("    Avg time:   %.3f ms\n", avg_ms);
        printf("    Alg BW:     %.2f GB/s  (bytes / time)\n", algbw_GBs);
        printf("    Bus BW:     %.2f GB/s  (2×(N-1)/N × alg BW)\n", bus_bw_GBs);
        printf("    Expected result per element: %.0f\n", expected);
        printf("    First 8 values: ");
        for (int i = 0; i < 8 && i < n_elems; ++i) printf("%.0f ", h_result[i]);
        printf("\n");
    }

    pthread_barrier_wait(&shared->barrier);
    printf("  GPU %d: %s (result[0]=%.0f, expected=%.0f)\n",
           rank, ok ? "PASS ✓" : "FAIL ✗", h_result[0], expected);

    // ── Cleanup ───────────────────────────────────────────────────
    free(h_result);
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(shared->comms[rank]));
    return nullptr;
}

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  ex12_nccl_allreduce — NCCL AllReduce End-to-End Demo\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    int n_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n_gpus));
    if (n_gpus < 2) {
        printf("Need >= 2 GPUs for NCCL test. Found %d. Exiting.\n", n_gpus);
        return 1;
    }
    printf("GPUs found: %d\n", n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, i));
        printf("  GPU %d: %s\n", i, p.name);
    }

    const int N_ELEMS = 8 * 1024 * 1024;  // 8M floats = 32 MB

    // ── Set up shared state ───────────────────────────────────────
    SharedState shared;
    shared.n_gpus = n_gpus;
    shared.comms  = new ncclComm_t[n_gpus];
    shared.rank_ok = new int[n_gpus]();
    pthread_barrier_init(&shared.barrier, nullptr, n_gpus);

    // ── Launch one thread per GPU ─────────────────────────────────
    pthread_t* threads = new pthread_t[n_gpus];
    ThreadArgs* args   = new ThreadArgs[n_gpus];

    for (int i = 0; i < n_gpus; ++i) {
        args[i] = {i, n_gpus, N_ELEMS, &shared};
        int status = pthread_create(&threads[i], nullptr, worker, &args[i]);
        if (status != 0) { fprintf(stderr, "pthread_create: %s\n", strerror(status)); exit(1); }
    }
    for (int i = 0; i < n_gpus; ++i) {
        int status = pthread_join(threads[i], nullptr);
        if (status != 0) { fprintf(stderr, "pthread_join: %s\n", strerror(status)); exit(1); }
    }

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  Expected sum = %d × (%d + 1) / 2 = %d\n",
           n_gpus, n_gpus, n_gpus * (n_gpus + 1) / 2);
    printf("═══════════════════════════════════════════════════════\n");

    bool all_ok = true;
    for (int i = 0; i < n_gpus; ++i) all_ok &= shared.rank_ok[i] == 1;
    printf("All-rank numerical verification: %s\n", all_ok ? "PASS" : "FAIL");
    delete[] shared.rank_ok;
    pthread_barrier_destroy(&shared.barrier);
    delete[] shared.comms;
    delete[] threads;
    delete[] args;
    return all_ok ? 0 : 1;
}
