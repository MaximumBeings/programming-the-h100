// ═══════════════════════════════════════════════════════════════════════
// ex13_hybrid_parallel_training.cu
//
// PURPOSE: Checked TP/PP/DP teaching harness with an independent CPU oracle.
//
// PARALLELISM:
//   TP=2  Tensor Parallelism: column + row linear layers split across 2 GPUs
//   PP=2  Pipeline Parallelism: 2 model stages, each on a different GPU pair
//   DP=2  Data Parallelism: 2 replicas, gradient AllReduce after each step
//   Total: 2 × 2 × 2 = 8 GPUs
//
// MODEL:
//   Stage 0: Column-parallel linear — X [B,H] × W0 [H, H/TP] = Y [B, H/TP]
//   Stage 1: Row-parallel linear    — Y [B, H/TP] × W1 [H/TP, H] = Z [B, H]
//            AllReduce(Z) to merge TP partial sums → final output
//   Loss: mean squared (target is zero) — loss = mean(Z^2)
//
// COMMUNICATORS:
//   tp_comm:  {GPU pair sharing the same PP stage and DP replica}
//             Used for: AllReduce of row-parallel output, AllGather if needed
//   pp_comm:  {GPU pair at consecutive PP stages, same TP rank, same DP rank}
//             Used for: ncclSend/ncclRecv of activations and gradients
//   dp_comm:  {GPU pair at same PP stage, same TP rank, different DP replica}
//             Used for: AllReduce of gradients across replicas
//
// TRAINING LOOP:
//   For each step:
//     1. Stage 0: column-parallel forward → ncclSend activation to stage 1
//     2. Stage 1: row-parallel forward → AllReduce (TP) → compute loss
//     3. Stage 1: backward through W1 → ncclSend grad back to stage 0
//     4. Stage 0: backward through W0
//     5. All: AllReduce gradients across DP replicas
//     6. All: SGD update with gradient clipping
//     7. Verify: all final weights and every per-replica step loss match CPU SGD
//
// WHAT TO LOOK FOR:
//   - Per-step losses match CPU reference (inputs change, so monotonicity is not assumed)
//   - DP replica weight divergence stays at exactly 0.0 (symmetry)
//   - This synchronized teaching loop does not claim production overlap
//   - TP shards reconstruct to the same result as single-GPU reference
//
// THREAD MODEL: Single process, 8 pthreads (one per GPU).
//   In production: one MPI rank per GPU, same NCCL API.
//
// WORKS ON: sm_80+, requires NCCL >= 2.15, cuBLAS.
// ═══════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <atomic>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>

// ── Error checking macros ─────────────────────────────────────────────────
#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[GPU %d] CUDA error %s:%d — %s\n", \
                g_gpu_id, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(call) do { \
    ncclResult_t _e = (call); \
    if (_e != ncclSuccess) { \
        fprintf(stderr, "[GPU %d] NCCL error %s:%d — %s\n", \
                g_gpu_id, __FILE__, __LINE__, ncclGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t _e = (call); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[GPU %d] cuBLAS error %s:%d — code %d\n", \
                g_gpu_id, __FILE__, __LINE__, (int)_e); \
        exit(1); \
    } \
} while(0)

static __thread int g_gpu_id = -1;   // set per thread for error macros

// ── Parallelism configuration ─────────────────────────────────────────────
constexpr int TP   = 2;   // tensor parallel degree
constexpr int PP   = 2;   // pipeline parallel stages
constexpr int DP   = 2;   // data parallel replicas
constexpr int NGPU = TP * PP * DP;   // = 8 GPUs total

// ── Model / training hyper-parameters ────────────────────────────────────
constexpr int HIDDEN    = 64;    // full hidden dimension
constexpr int BATCH     = 16;    // micro-batch size (tokens)
constexpr int N_STEPS   = 8;     // training steps to run
constexpr float LR      = 0.01f; // SGD learning rate
constexpr float CLIP    = 1.0f;  // gradient clip norm

// Derived sizes
constexpr int H_SHARD = HIDDEN / TP;   // per-TP-rank weight width = 32

// ── GPU rank → parallelism coordinates ───────────────────────────────────
//
//  Global GPU numbering layout (row = PP stage, col = TP rank):
//
//  DP replica 0              DP replica 1
//  pp=0: gpu 0 (tp=0)  gpu 1 (tp=1)   gpu 4 (tp=0)  gpu 5 (tp=1)
//  pp=1: gpu 2 (tp=0)  gpu 3 (tp=1)   gpu 6 (tp=0)  gpu 7 (tp=1)
//
//  Formula: gpu_id = dp_rank*(TP*PP) + pp_rank*TP + tp_rank
//
inline int gpu_of(int dp, int pp, int tp) {
    return dp * (TP * PP) + pp * TP + tp;
}
inline int tp_rank_of(int gpu)  { return gpu % TP; }
inline int pp_rank_of(int gpu)  { return (gpu / TP) % PP; }
inline int dp_rank_of(int gpu)  { return gpu / (TP * PP); }

// ── Shared initialization state ───────────────────────────────────────────
struct SharedState {
    // NCCL unique IDs for three communicators
    ncclUniqueId tp_ids[PP * DP];  // one TP comm per {pp, dp} group
    ncclUniqueId pp_ids[TP * DP];  // one PP comm per {tp, dp} column
    ncclUniqueId dp_ids[TP * PP];  // one DP comm per {tp, pp} position

    // Per-GPU communicators (indexed by GPU id)
    ncclComm_t   tp_comm[NGPU];
    ncclComm_t   pp_comm[NGPU];
    ncclComm_t   dp_comm[NGPU];

    // Barrier for all-thread synchronization
    pthread_barrier_t barrier;

    // Loss values for monitoring (indexed by pp_rank, dp_rank)
    float losses[PP][DP][N_STEPS];
    pthread_mutex_t loss_mutex;
    bool rank_ok[NGPU] = {};
    float reference_weights[PP][HIDDEN * HIDDEN];
    float reference_losses[DP][N_STEPS];
};

// Host-only reference: full, unsharded row-major matrices and scalar loops.
// Inputs and initial weights are shared definitions; no cuBLAS/NCCL path is reused.
float initial_weight(int stage, int row, int col) {
    int index = stage * HIDDEN * HIDDEN + row * HIDDEN + col + 1;
    return 0.05f * sinf(index * 0.1f);
}
float training_input(int step, int dp, int i) {
    return sinf((float)step + dp * 0.17f + i * 0.01f);
}
void compute_cpu_reference(SharedState* shared) {
    const int H = HIDDEN, B = BATCH;
    for (int stage = 0; stage < 2; ++stage)
        for (int r = 0; r < H; ++r)
            for (int col = 0; col < H; ++col)
                shared->reference_weights[stage][r * H + col] = initial_weight(stage, r, col);
    std::vector<double> x(B*H), y(B*H), z(B*H), dz(B*H), dy(B*H);
    std::vector<double> mean_g0(H*H), mean_g1(H*H);
    for (int step = 0; step < N_STEPS; ++step) {
        std::fill(mean_g0.begin(), mean_g0.end(), 0.0);
        std::fill(mean_g1.begin(), mean_g1.end(), 0.0);
        const float* w0 = shared->reference_weights[0];
        const float* w1 = shared->reference_weights[1];
        for (int dp = 0; dp < DP; ++dp) {
            for (int i = 0; i < B*H; ++i) x[i] = training_input(step, dp, i);
            // Y = X W0; Z = Y W1. Reference accumulation is double precision.
            for (int b = 0; b < B; ++b)
                for (int j = 0; j < H; ++j) {
                    double sum = 0;
                    for (int k = 0; k < H; ++k) sum += x[b*H+k] * w0[k*H+j];
                    y[b*H+j] = sum;
                }
            double loss = 0;
            for (int b = 0; b < B; ++b)
                for (int j = 0; j < H; ++j) {
                    double sum = 0;
                    for (int k = 0; k < H; ++k) sum += y[b*H+k] * w1[k*H+j];
                    z[b*H+j] = sum;
                    loss += sum * sum;
                    dz[b*H+j] = 2.0 * sum / (B*H);
                }
            shared->reference_losses[dp][step] = (float)(loss / (B*H));
            // dY = dZ W1^T; dW1 = Y^T dZ; dW0 = X^T dY.
            for (int b = 0; b < B; ++b)
                for (int k = 0; k < H; ++k) {
                    double sum = 0;
                    for (int j = 0; j < H; ++j) sum += dz[b*H+j] * w1[k*H+j];
                    dy[b*H+k] = sum;
                }
            for (int k = 0; k < H; ++k)
                for (int j = 0; j < H; ++j) {
                    double g0 = 0, g1 = 0;
                    for (int b = 0; b < B; ++b) {
                        g0 += x[b*H+k] * dy[b*H+j];
                        g1 += y[b*H+k] * dz[b*H+j];
                    }
                    mean_g0[k*H+j] += g0 / DP;
                    mean_g1[k*H+j] += g1 / DP;
                }
        }
        // Match the implementation's clipping of EACH TP weight shard after
        // DP averaging: W0 column shards, W1 row shards. This is not global clipping.
        for (int stage = 0; stage < 2; ++stage) {
            const auto& gradient = stage == 0 ? mean_g0 : mean_g1;
            for (int tp = 0; tp < TP; ++tp) {
                double norm_sq = 0;
                for (int r = 0; r < H; ++r)
                    for (int col = 0; col < H; ++col) {
                        if ((stage == 0 ? col / H_SHARD : r / H_SHARD) == tp) {
                            double g = gradient[r*H+col];
                            norm_sq += g*g;
                        }
                    }
                double norm = std::sqrt(norm_sq);
                double scale = norm > CLIP ? CLIP / norm : 1.0;
                for (int r = 0; r < H; ++r)
                    for (int col = 0; col < H; ++col)
                        if ((stage == 0 ? col / H_SHARD : r / H_SHARD) == tp) {
                            int i = r*H+col;
                            shared->reference_weights[stage][i] = (float)
                                (shared->reference_weights[stage][i] - (double)LR * scale * gradient[i]);
                        }
            }
        }
    }
}

// ── Per-GPU training state ────────────────────────────────────────────────
struct GpuState {
    int gpu_id, tp_rank, pp_rank, dp_rank;

    cublasHandle_t cublas;
    cudaStream_t   stream_compute;   // matmul, local ops
    cudaStream_t   stream_comm;      // NCCL AllReduce overlap

    // Weight and gradient buffers (column or row shard depending on PP stage)
    // Stage 0 (pp=0): column-parallel W0 of shape [HIDDEN, H_SHARD]
    // Stage 1 (pp=1): row-parallel    W1 of shape [H_SHARD, HIDDEN]
    float *d_weight;   // [HIDDEN × H_SHARD] for stage 0
                       // [H_SHARD × HIDDEN] for stage 1
    float *d_grad;     // same shape as d_weight

    // Activation buffers
    float *d_input;         // [BATCH, HIDDEN]   — local input/activation
    float *d_output;        // [BATCH, H_SHARD]  — stage 0 output (pre-gather)
                            // [BATCH, HIDDEN]   — stage 1 output (post-allreduce)
    float *d_grad_output;   // gradient w.r.t. output (received from next stage or loss)
    float *d_grad_input;    // gradient w.r.t. input (to send back to prev stage)

    // Scratchpad for partial output before TP AllReduce
    float *d_partial;       // [BATCH, HIDDEN] — stage 1 partial sum before AllReduce

    // Norm for gradient clipping
    float *d_gnorm_sq;      // [1] — sum of squared gradients
};

// ── CUDA kernels ──────────────────────────────────────────────────────────

// Fill buffer with a constant value
__global__ void fill_kernel(float* buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

// MSE loss gradient: dL/dZ = 2*Z / N  (target = 0)
__global__ void mse_grad_kernel(const float* Z, float* dZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dZ[i] = 2.0f * Z[i] / n;
}

// SGD update with gradient clipping: W -= lr * grad / max(1, gnorm/clip)
__global__ void sgd_clip_kernel(float* W, const float* grad,
                                 const float* gnorm_sq,
                                 float lr, float clip, int n) {
    float gnorm = sqrtf(*gnorm_sq);
    float scale = (gnorm > clip) ? clip / gnorm : 1.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) W[i] -= lr * scale * grad[i];
}

// Reduce sum of squared elements into a single float (for gradient norm)
// Simple single-block reduction (fine for small weight tensors here)
__global__ void sum_sq_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ── Helper: cuBLAS SGEMM wrapper ─────────────────────────────────────────
// C = alpha * A × B + beta * C
// A: [M, K], B: [K, N], C: [M, N]  (row-major stored)
// cuBLAS is column-major, so we compute C^T = B^T × A^T
void sgemm(cublasHandle_t h, int M, int N, int K,
           const float* A, const float* B, float* C,
           float alpha = 1.0f, float beta = 0.0f) {
    // We want C[M,N] = A[M,K] × B[K,N]
    // In column-major: C^T[N,M] = B^T[N,K] × A^T[K,M]
    CHECK_CUBLAS(cublasSgemm(h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,       // (n, m, k) in col-major
        &alpha,
        B, N,          // B^T: leading dim = N
        A, K,          // A^T: leading dim = K
        &beta,
        C, N));        // C^T: leading dim = N
}

// ── Stage 0 forward: column-parallel linear ──────────────────────────────
// X [BATCH, HIDDEN] × W0 [HIDDEN, H_SHARD] = Y [BATCH, H_SHARD]
// Input is replicated; each TP rank computes its shard of the output.
// No communication needed during forward (communication happens at boundary).
void stage0_forward(GpuState& s) {
    // d_input [BATCH, HIDDEN], d_weight [HIDDEN, H_SHARD] → d_output [BATCH, H_SHARD]
    sgemm(s.cublas, BATCH, H_SHARD, HIDDEN,
          s.d_input, s.d_weight, s.d_output);
}

// ── Stage 0 backward ─────────────────────────────────────────────────────
// Received: d_grad_output [BATCH, H_SHARD]  (from stage 1 via PP send)
// Compute:  d_grad [HIDDEN, H_SHARD] = input^T × d_grad_output
//           d_grad_input [BATCH, HIDDEN] = d_grad_output × W0^T
// dW0 = X^T [HIDDEN, BATCH] × dY [BATCH, H_SHARD] = [HIDDEN, H_SHARD]
// dX  = dY  [BATCH, H_SHARD] × W0^T [H_SHARD, HIDDEN] = [BATCH, HIDDEN]
void stage0_backward(GpuState& s) {
    // dW0 = X^T dY, with explicit transpose flags for row-major storage.
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(s.cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,   // dY, X^T
        H_SHARD, HIDDEN, BATCH,
        &alpha,
        s.d_grad_output, H_SHARD,   // dY [BATCH, H_SHARD]
        s.d_input,       HIDDEN,    // X  [BATCH, HIDDEN]  → X^T
        &beta,
        s.d_grad,        H_SHARD)); // dW [HIDDEN, H_SHARD]

    // Gradient w.r.t. input (for layers before stage 0, here just computed)
    CHECK_CUBLAS(cublasSgemm(s.cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,   // W0^T, dY
        HIDDEN, BATCH, H_SHARD,
        &alpha,
        s.d_weight,      H_SHARD,   // W0 [HIDDEN, H_SHARD] → W0^T
        s.d_grad_output, H_SHARD,   // dY [BATCH, H_SHARD]
        &beta,
        s.d_grad_input,  HIDDEN));  // dX [BATCH, HIDDEN]
}

// ── Stage 1 forward: row-parallel linear ─────────────────────────────────
// Y_shard [BATCH, H_SHARD] × W1 [H_SHARD, HIDDEN] = Z_partial [BATCH, HIDDEN]
// AllReduce Z_partial across TP group → Z [BATCH, HIDDEN]  (sum of row-partial results)
void stage1_forward(GpuState& s, ncclComm_t tp_comm) {
    // d_input [BATCH, H_SHARD], d_weight [H_SHARD, HIDDEN] → d_partial [BATCH, HIDDEN]
    sgemm(s.cublas, BATCH, HIDDEN, H_SHARD,
          s.d_input, s.d_weight, s.d_partial);

    // TP AllReduce: sum partial results from both TP ranks
    // After this, d_output = full Z = Y × W1 (complete, not sharded)
    CHECK_NCCL(ncclAllReduce(s.d_partial, s.d_output,
                              BATCH * HIDDEN, ncclFloat, ncclSum,
                              tp_comm, s.stream_compute));
    CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));
}

// ── Stage 1 backward ─────────────────────────────────────────────────────
// Loss gradient dL/dZ [BATCH, HIDDEN] flows back through row-parallel W1.
// dW1 = Y_shard^T × dZ  [H_SHARD, HIDDEN]
// dY_shard = dZ × W1^T  [BATCH, H_SHARD]  (sent to stage 0 via PP)
void stage1_backward(GpuState& s) {
    float alpha = 1.0f, beta = 0.0f;

    // dW1 [H_SHARD, HIDDEN] = input_shard^T [H_SHARD, BATCH] × dZ [BATCH, HIDDEN]
    CHECK_CUBLAS(cublasSgemm(s.cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        HIDDEN, H_SHARD, BATCH,
        &alpha,
        s.d_grad_output, HIDDEN,   // dZ [BATCH, HIDDEN]
        s.d_input,       H_SHARD,  // input_shard [BATCH, H_SHARD] → transposed
        &beta,
        s.d_grad,        HIDDEN)); // dW1 [H_SHARD, HIDDEN]

    // dY_shard [BATCH, H_SHARD] = dZ [BATCH, HIDDEN] × W1^T [HIDDEN, H_SHARD]
    CHECK_CUBLAS(cublasSgemm(s.cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        H_SHARD, BATCH, HIDDEN,
        &alpha,
        s.d_weight,      HIDDEN,   // W1 [H_SHARD, HIDDEN] → W1^T [HIDDEN, H_SHARD]
        s.d_grad_output, HIDDEN,   // dZ [BATCH, HIDDEN]
        &beta,
        s.d_grad_input,  H_SHARD));// dY_shard [BATCH, H_SHARD]
}

// ── Worker thread: one per GPU ────────────────────────────────────────────
void* worker(void* arg_ptr) {
    SharedState* shared = (SharedState*)arg_ptr;
    // Extract this thread's GPU ID from a global counter
    static std::atomic<int> gpu_counter{0};
    int gpu_id = gpu_counter.fetch_add(1);
    g_gpu_id   = gpu_id;

    int tp = tp_rank_of(gpu_id);
    int pp = pp_rank_of(gpu_id);
    int dp = dp_rank_of(gpu_id);

    CHECK_CUDA(cudaSetDevice(gpu_id));

    // ── Phase 1: Build communicators ─────────────────────────────────
    // TP communicator: all GPUs with same {pp, dp} — differ only in tp_rank
    {
        int group_id = dp * PP + pp;   // unique ID for this TP group
        if (tp == 0) {
            // First TP rank generates the unique ID for this group
            CHECK_NCCL(ncclGetUniqueId(&shared->tp_ids[group_id]));
        }
        pthread_barrier_wait(&shared->barrier);
        CHECK_NCCL(ncclCommInitRank(&shared->tp_comm[gpu_id],
                                     TP,
                                     shared->tp_ids[group_id],
                                     tp));
    }

    // PP communicator: all GPUs with same {tp, dp} — differ only in pp_rank
    {
        int group_id = dp * TP + tp;
        if (pp == 0) {
            CHECK_NCCL(ncclGetUniqueId(&shared->pp_ids[group_id]));
        }
        pthread_barrier_wait(&shared->barrier);
        CHECK_NCCL(ncclCommInitRank(&shared->pp_comm[gpu_id],
                                     PP,
                                     shared->pp_ids[group_id],
                                     pp));
    }

    // DP communicator: all GPUs with same {tp, pp} — differ only in dp_rank
    {
        int group_id = pp * TP + tp;
        if (dp == 0) {
            CHECK_NCCL(ncclGetUniqueId(&shared->dp_ids[group_id]));
        }
        pthread_barrier_wait(&shared->barrier);
        CHECK_NCCL(ncclCommInitRank(&shared->dp_comm[gpu_id],
                                     DP,
                                     shared->dp_ids[group_id],
                                     dp));
    }
    pthread_barrier_wait(&shared->barrier);

    // ── Phase 2: Allocate GPU buffers ────────────────────────────────
    GpuState s;
    s.gpu_id  = gpu_id;
    s.tp_rank = tp;
    s.pp_rank = pp;
    s.dp_rank = dp;

    CHECK_CUDA(cudaStreamCreate(&s.stream_compute));
    CHECK_CUDA(cudaStreamCreate(&s.stream_comm));
    CHECK_CUBLAS(cublasCreate(&s.cublas));
    CHECK_CUBLAS(cublasSetStream(s.cublas, s.stream_compute));
    CHECK_CUBLAS(cublasSetMathMode(s.cublas, CUBLAS_PEDANTIC_MATH));

    int weight_elems = (pp == 0) ? (HIDDEN * H_SHARD)   // W0: [H, H/TP]
                                  : (H_SHARD * HIDDEN);  // W1: [H/TP, H]
    int output_cols  = (pp == 0) ? H_SHARD : HIDDEN;

    CHECK_CUDA(cudaMalloc(&s.d_weight,      weight_elems      * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_grad,        weight_elems      * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_input,       BATCH * HIDDEN    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_output,      BATCH * HIDDEN    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_partial,     BATCH * HIDDEN    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_grad_output, BATCH * HIDDEN    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_grad_input,  BATCH * HIDDEN    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_gnorm_sq,    1                 * sizeof(float)));

    // ── Phase 3: Initialize weights ──────────────────────────────────
    // Balanced, deterministic full-matrix initialization; both DP replicas agree.
    {
        float* h_w = new float[weight_elems];
        for (int i = 0; i < weight_elems; ++i) {
            int row = pp == 0 ? i / H_SHARD : tp * H_SHARD + i / HIDDEN;
            int col = pp == 0 ? tp * H_SHARD + i % H_SHARD : i % HIDDEN;
            h_w[i] = initial_weight(pp, row, col);
        }
        CHECK_CUDA(cudaMemcpy(s.d_weight, h_w, weight_elems * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_w;
    }

    // ── Phase 4: Training loop ────────────────────────────────────────
    float loss_this_gpu = 0.0f;

    for (int step = 0; step < N_STEPS; ++step) {

        // ── 4a. Generate input data ───────────────────────────────────
        // Distinct DP inputs make gradient averaging necessary for oracle agreement.
        // PP stage 0 generates its own input.
        // PP stage 1 receives activations via PP send/recv.
        if (pp == 0) {
            float* h_inp = new float[BATCH * HIDDEN];
            for (int i = 0; i < BATCH * HIDDEN; ++i)
                h_inp[i] = training_input(step, dp, i);
            CHECK_CUDA(cudaMemcpyAsync(s.d_input, h_inp,
                                       BATCH * HIDDEN * sizeof(float),
                                       cudaMemcpyHostToDevice, s.stream_compute));
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));
            delete[] h_inp;
        }

        // ── 4b. FORWARD PASS ─────────────────────────────────────────

        if (pp == 0) {
            // Stage 0: column-parallel forward
            stage0_forward(s);
            // s.d_output is now [BATCH, H_SHARD] (this TP rank's output shard)

            // PP send: send output shard to stage 1 (same tp_rank, same dp_rank)
            int dst_gpu = gpu_of(dp, /*pp=*/1, tp);
            // Use pp_comm (both pp=0 and pp=1 are in the same pp communicator)
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(s.d_output,
                                 BATCH * H_SHARD, ncclFloat,
                                 /*peer=*/1,   // pp=1 rank in pp_comm
                                 shared->pp_comm[gpu_id],
                                 s.stream_compute));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        } else {
            // Stage 1: receive activation shard from stage 0
            // d_input here represents Y_shard [BATCH, H_SHARD]
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclRecv(s.d_input,
                                 BATCH * H_SHARD, ncclFloat,
                                 /*peer=*/0,   // pp=0 rank in pp_comm
                                 shared->pp_comm[gpu_id],
                                 s.stream_compute));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

            // Stage 1: row-parallel forward + TP AllReduce
            stage1_forward(s, shared->tp_comm[gpu_id]);
            // s.d_output is now [BATCH, HIDDEN] (full output after AllReduce)

            // Compute loss (only on stage 1, tp_rank 0 for clean print)
            if (tp == 0) {
                float* h_out = new float[BATCH * HIDDEN];
                CHECK_CUDA(cudaMemcpy(h_out, s.d_output,
                                      BATCH * HIDDEN * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                float loss = 0.0f;
                for (int i = 0; i < BATCH * HIDDEN; ++i)
                    loss += h_out[i] * h_out[i];
                loss /= (BATCH * HIDDEN);
                loss_this_gpu = loss;
                shared->losses[pp][dp][step] = loss;
                delete[] h_out;
            }

            // Compute loss gradient: dL/dZ = 2Z / N
            int n = BATCH * HIDDEN;
            mse_grad_kernel<<<(n+255)/256, 256, 0, s.stream_compute>>>(
                s.d_output, s.d_grad_output, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));
        }

        pthread_barrier_wait(&shared->barrier);   // sync PP stages before backward

        // ── 4c. BACKWARD PASS ─────────────────────────────────────────

        if (pp == 1) {
            // Stage 1 backward
            stage1_backward(s);
            // s.d_grad_input [BATCH, H_SHARD] = dY_shard → send to stage 0

            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(s.d_grad_input,
                                 BATCH * H_SHARD, ncclFloat,
                                 /*peer=*/0,
                                 shared->pp_comm[gpu_id],
                                 s.stream_compute));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        } else {
            // Stage 0: receive gradient from stage 1
            // s.d_grad_output [BATCH, H_SHARD] = dY_shard received
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclRecv(s.d_grad_output,
                                 BATCH * H_SHARD, ncclFloat,
                                 /*peer=*/1,
                                 shared->pp_comm[gpu_id],
                                 s.stream_compute));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

            // Stage 0 backward
            stage0_backward(s);
            // s.d_grad [HIDDEN, H_SHARD] = dW0
        }

        pthread_barrier_wait(&shared->barrier);   // all stages done with backward

        // ── 4d. DP AllReduce: average gradients across DP replicas ────
        // On a separate stream to enable compute/comm overlap (here simplified
        // since we have no next-layer compute to overlap; see comment below).
        //
        // In production: launch the AllReduce on stream_comm WHILE the next
        // micro-batch's forward pass runs on stream_compute. Use cudaEvent
        // to make the optimizer wait for stream_comm:
        //   cudaEvent_t e; cudaEventRecord(e, stream_comm);
        //   cudaStreamWaitEvent(stream_optimizer, e, 0);
        //
        // Here we use stream_compute for simplicity.
        CHECK_NCCL(ncclAllReduce(s.d_grad, s.d_grad,
                                  weight_elems, ncclFloat, ncclSum,
                                  shared->dp_comm[gpu_id],
                                  s.stream_compute));
        CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        // Average: divide by DP (ncclSum gives sum, not mean)
        float inv_dp = 1.0f / DP;
        CHECK_CUBLAS(cublasSscal(s.cublas, weight_elems, &inv_dp, s.d_grad, 1));
        CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        // ── 4e. Gradient clipping ─────────────────────────────────────
        CHECK_CUDA(cudaMemsetAsync(s.d_gnorm_sq, 0, sizeof(float), s.stream_compute));
        int block = 256;
        int grid  = (weight_elems + block - 1) / block;
        sum_sq_kernel<<<grid, block, block*sizeof(float), s.stream_compute>>>(
            s.d_grad, s.d_gnorm_sq, weight_elems);
        CHECK_CUDA(cudaGetLastError());

        // ── 4f. SGD update ────────────────────────────────────────────
        sgd_clip_kernel<<<grid, block, 0, s.stream_compute>>>(
            s.d_weight, s.d_grad, s.d_gnorm_sq, LR, CLIP, weight_elems);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        // Zero gradients for next step
        CHECK_CUDA(cudaMemsetAsync(s.d_grad, 0, weight_elems * sizeof(float),
                                   s.stream_compute));
        CHECK_CUDA(cudaStreamSynchronize(s.stream_compute));

        pthread_barrier_wait(&shared->barrier);   // step complete

        // ── 4g. Log loss (rank 0 of each DP replica's stage 1, tp=0) ─
        if (tp == 0 && pp == 1) {
            printf("  Step %2d | DP replica %d | loss = %.6f\n",
                   step, dp, shared->losses[pp][dp][step]);
        }
        pthread_barrier_wait(&shared->barrier);
    }

    // ── Phase 5: Verification ─────────────────────────────────────────────

    // 5a. Check DP replicas have identical weights
    // Each GPU copies its weight to host. GPUs with same {tp,pp} but different
    // dp_rank should have IDENTICAL weights after training (DP symmetry).
    float* h_weight_local  = new float[weight_elems];
    float* h_weight_peer   = new float[weight_elems];
    CHECK_CUDA(cudaMemcpy(h_weight_local, s.d_weight,
                          weight_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Share weights across DP replicas via AllGather into a temporary
    // We compare: dp=0's weight vs dp=1's weight for same {tp,pp}
    // Simple approach: each rank posts its weight into a shared buffer
    static float g_weights[NGPU][4096];   // static: visible to all threads
    memcpy(g_weights[gpu_id], h_weight_local, weight_elems * sizeof(float));
    pthread_barrier_wait(&shared->barrier);

    bool dp_ok = true;
    for (int i = 0; i < weight_elems; ++i)
        if (!std::isfinite(h_weight_local[i])) dp_ok = false;
    if (dp == 0) {
        int peer_gpu = gpu_of(/*dp=*/1, pp, tp);
        float max_diff = 0.0f;
        for (int i = 0; i < weight_elems; ++i) {
            if (!std::isfinite(g_weights[gpu_id][i]) ||
                !std::isfinite(g_weights[peer_gpu][i])) dp_ok = false;
            float diff = fabsf(g_weights[gpu_id][i] - g_weights[peer_gpu][i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-5f) dp_ok = false;
        printf("  [Verify] GPU %d vs GPU %d (DP replica diff): max_diff = %.2e  %s\n",
               gpu_id, peer_gpu, max_diff, dp_ok ? "PASS ✓" : "FAIL ✗");
    }
    pthread_barrier_wait(&shared->barrier);

    // 5b. Verify every local shard element against the independent full CPU model.
    bool weight_ok = true;
    float max_weight_error = 0.0f;
    for (int i = 0; i < weight_elems; ++i) {
        int row = pp == 0 ? i / H_SHARD : tp * H_SHARD + i / HIDDEN;
        int col = pp == 0 ? tp * H_SHARD + i % H_SHARD : i % HIDDEN;
        float expected = shared->reference_weights[pp][row * HIDDEN + col];
        float got = h_weight_local[i];
        float error = fabsf(got - expected);
        max_weight_error = std::max(max_weight_error, error);
        if (!std::isfinite(got) || !std::isfinite(expected) ||
            error > 1e-6f + 1e-4f * fabsf(expected)) weight_ok = false;
    }
    printf("  [Verify] GPU %d CPU final-weight reference: %s; elements=%d max_abs_error=%.9g\n",
           gpu_id, weight_ok ? "PASS" : "FAIL", weight_elems, max_weight_error);

    // 5c. Inputs change each step: compare all losses to the oracle, not to
    // the previous step. Distinct DP inputs also test the DP gradient average.
    bool loss_ok = true;
    if (pp == 1 && tp == 0) {
        float max_loss_error = 0.0f;
        for (int step = 0; step < N_STEPS; ++step) {
            float got = shared->losses[1][dp][step];
            float expected = shared->reference_losses[dp][step];
            float error = fabsf(got - expected);
            max_loss_error = std::max(max_loss_error, error);
            if (!std::isfinite(got) || !std::isfinite(expected) ||
                error > 1e-6f + 1e-4f * fabsf(expected)) {
                loss_ok = false;
                printf("  CPU loss mismatch dp=%d step=%d got=%.9g expected=%.9g\n",
                       dp, step, got, expected);
            }
        }
        printf("  [Verify] DP replica %d per-step CPU loss reference: %s; steps=%d max_abs_error=%.9g\n",
               dp, loss_ok ? "PASS" : "FAIL", N_STEPS, max_loss_error);
    }
    pthread_barrier_wait(&shared->barrier);
    shared->rank_ok[gpu_id] = dp_ok && weight_ok && loss_ok;

    // ── Phase 6: Cleanup ─────────────────────────────────────────────────
    delete[] h_weight_local;
    delete[] h_weight_peer;
    CHECK_CUDA(cudaFree(s.d_weight));
    CHECK_CUDA(cudaFree(s.d_grad));
    CHECK_CUDA(cudaFree(s.d_input));
    CHECK_CUDA(cudaFree(s.d_output));
    CHECK_CUDA(cudaFree(s.d_partial));
    CHECK_CUDA(cudaFree(s.d_grad_output));
    CHECK_CUDA(cudaFree(s.d_grad_input));
    CHECK_CUDA(cudaFree(s.d_gnorm_sq));
    CHECK_CUBLAS(cublasDestroy(s.cublas));
    CHECK_CUDA(cudaStreamDestroy(s.stream_compute));
    CHECK_CUDA(cudaStreamDestroy(s.stream_comm));
    CHECK_NCCL(ncclCommDestroy(shared->tp_comm[gpu_id]));
    CHECK_NCCL(ncclCommDestroy(shared->pp_comm[gpu_id]));
    CHECK_NCCL(ncclCommDestroy(shared->dp_comm[gpu_id]));
    return nullptr;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  ex13_hybrid_parallel_training\n");
    printf("  TP=%d × PP=%d × DP=%d = %d GPUs\n", TP, PP, DP, NGPU);
    printf("  Hidden=%d  Batch=%d  Steps=%d  LR=%.3f\n",
           HIDDEN, BATCH, N_STEPS, LR);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Check GPU count
    // (See ex12's comment on the same pattern: initialize explicitly,
    // since cudaGetDeviceCount's return code isn't checked here and
    // an uninitialized n_gpus can read back as garbage -- reproduced
    // live as "found -1" when this exact uninitialized version ran
    // with zero CUDA-capable devices present.)
    int n_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n_gpus));
    if (n_gpus < NGPU) {
        printf("ERROR: Need %d GPUs, found %d. Exiting.\n", NGPU, n_gpus);
        return 1;
    }

    // Print GPU layout
    printf("GPU layout:\n");
    for (int d = 0; d < DP; ++d) {
        printf("  DP replica %d:\n", d);
        for (int p = 0; p < PP; ++p) {
            printf("    PP stage %d: ", p);
            for (int t = 0; t < TP; ++t) {
                printf("GPU %d (tp=%d)  ", gpu_of(d, p, t), t);
            }
            printf("\n");
        }
    }
    printf("\n");

    // Set up shared state
    SharedState* shared = new SharedState();
    memset(shared->losses, 0, sizeof(shared->losses));
    compute_cpu_reference(shared);
    printf("CPU serial reference prepared: full matrices, distinct DP inputs, per-shard clipping.\n");
    pthread_barrier_init(&shared->barrier, nullptr, NGPU);
    pthread_mutex_init(&shared->loss_mutex, nullptr);

    // Launch threads
    printf("Training:\n");
    pthread_t threads[NGPU];
    for (int i = 0; i < NGPU; ++i) {
        if (pthread_create(&threads[i], nullptr, worker, shared) != 0) {
            fprintf(stderr, "pthread_create failed\n"); exit(1);
        }
    }
    for (int i = 0; i < NGPU; ++i) {
        if (pthread_join(threads[i], nullptr) != 0) {
            fprintf(stderr, "pthread_join failed\n"); exit(1);
        }
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  Training complete. See verification output above.\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    bool all_ok = true;
    for (int i = 0; i < NGPU; ++i) all_ok = all_ok && shared->rank_ok[i];
    printf("All-rank training validation: %s\n", all_ok ? "PASS" : "FAIL");
    pthread_barrier_destroy(&shared->barrier);
    pthread_mutex_destroy(&shared->loss_mutex);
    delete shared;
    return all_ok ? 0 : 1;
}
