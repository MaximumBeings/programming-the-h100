# Programming the H100

## A hardware-level companion guide to NVIDIA Hopper

<p class="hero-tagline">
The gap between "my kernel runs" and "my kernel saturates the tensor cores" is
exactly the size of this book. Thirteen chapters that start at the H100 die and
end with a profiler telling you which of your assumptions was wrong — every one
with ASCII diagrams, worked examples with concrete numbers, and code you can
read line by line.
</p>

> **Verified on:** CUDA 12.x + Hopper GPU (`sm_90a`).
> **Compile CUDA examples with:** `nvcc -std=c++17 -arch=sm_90a -O2 <file>.cu -o <out>`
> **Run with:** `./<out>` on any H100 or B200 instance.

## How to read this

The chapters build on each other. Chapter 1 is pure architecture — no code,
just the machine. Chapters 2–3 give you the instruction-level vocabulary
(clusters, inline PTX, state spaces, `mbarrier`). Chapters 4–5 are about
getting bytes into shared memory efficiently. Chapters 6–7 are the tensor
cores. Chapters 8–10 put it together into real warp-specialized kernels.
Chapter 11 leaves the single GPU behind. Chapters 12–13 are the payoff: one
production kernel that uses everything at once, and the tool that measures it.

If you already write CUDA and want the Hopper-specific parts, start at
Chapter 2. If you are here for GEMM, read 4 → 5 → 6 → 7 → 8 → 9 in order. If you
have a slow kernel in front of you right now, start at Chapter 13.

## Contents

<div class="grid cards" markdown>

-   :material-chip:{ .lg .middle } __Part I — The Machine__

    ---

    **[1. NVIDIA H100 GPU Architecture](01-h100-architecture.md)**

    The TMA unit, asynchronous execution, 4th-gen Tensor Cores, the memory
    hierarchy from HBM3 through L2/L1 to registers, SM microarchitecture
    (quadrants, warp schedulers, dispatch units, SFUs, LSUs), NVLink 4.0,
    shared memory banking and swizzling, and how thread blocks are
    distributed across SMs.

-   :material-code-braces:{ .lg .middle } __Part II — The Instruction Layer__

    ---

    **[2. Clusters, Inline PTX, Data Types & Pointers](02-clusters-inline-ptx.md)**

    Thread block clusters, distributed shared memory, TMA multicast, the PTX
    ISA, inline `asm()` syntax with operands/constraints/clobbers, state
    spaces, deep-learning data types, byte addressing, shared memory banks,
    `cvta` and `mapa`.

    **[3. Asynchronicity, Barriers & the mbarrier Pipeline](03-asynchronicity-mbarrier.md)**

    Latency hiding, blocking vs non-blocking operations, the Generic vs Async
    proxy memory model, RAW/WAR hazards, fences, the `mbarrier` primitive end
    to end, phase flipping, `barrier.cluster`, commit/wait groups, and named
    barriers.

-   :material-transfer:{ .lg .middle } __Part III — Moving Data__

    ---

    **[4. cuTensorMap, TMA & Swizzling](04-cutensormap-tma-swizzling.md)**

    What TMA is and why it exists, how descriptors replace manual address
    math, all 12 parameters of `cuTensorMapEncodeTiled()`, tensor rank,
    strides and box dimensions, 32B/64B/128B swizzling with the XOR formula,
    interleaving, L2 promotion, and out-of-bounds fill.

    **[5. cp.async.bulk, TMA Transfers & L2 Cache Hinting](05-cp-async-bulk-l2-hinting.md)**

    The `cp.async.bulk` family vs Ampere's `cp.async`, tensor-aware vs raw
    copies, every state-space combination, mbarrier vs bulk-group completion,
    L2 eviction hints and `createpolicy`, TMA multicast and the `ctaMask`,
    prefetching, and `cp.reduce.async`.

-   :material-matrix:{ .lg .middle } __Part IV — Tensor Cores__

    ---

    **[6. WGMMA — Warp Group Matrix Multiply-Accumulate](06-wgmma-part-1.md)**

    The shift from warp to warp group, why `wgmma.mma_async` is non-blocking,
    the fence → load → compute → collect pipeline, every qualifier, operand
    location rules, `ldmatrix` and swizzle addressing, the 64-bit WGMMA
    descriptor field by field, and all supported tile shapes.

    **[7. WGMMA Part 2 — Groups, stmatrix, FP8 & Sparse](07-wgmma-part-2.md)**

    Commit/wait group mechanics and what actually overlaps, the dual role of
    `wgmma.fence`, `stmatrix` writeback, the FP8 paradigm shift (E4M3 vs
    E5M2, scaling, `cvt.satfinite`, K-major layout, precision pitfalls), and
    sparse WGMMA with 2:4 structured sparsity.

-   :material-cog-play:{ .lg .middle } __Part V — Kernel Design__

    ---

    **[8. Kernel Design, Warp Specialization & Pipelining](08-kernel-design-part-1.md)**

    The roofline model and arithmetic intensity, classifying compute-bound vs
    memory-bound kernels, producer/consumer warp specialization,
    `setmaxnreg` dynamic register allocation, warp group count selection, and
    circular multi-stage buffering.

    **[9. Pipelines, Epilogues & Persistent Scheduling](09-kernel-design-part-2.md)**

    The two-barrier handshake, ABA ambiguity and phase bits, full producer and
    consumer flows, cooperative vs ping-pong pipeline architectures, the
    six-stage epilogue fusion lifecycle, persistent scheduling, and
    rasterization for L2 locality.

    **[10. Stream-K Scheduling & Multi-Kernel Launch](10-stream-k-scheduling.md)**

    Stream-K work units and splits, the first/middle/final triplet, the lock
    array, deterministic vs non-deterministic reduction, HyTiS, the stream
    serialization utilization gap, GDC hardware grid handoff, programmatic
    dependent launch, and tuning the overlap window.

-   :material-lan:{ .lg .middle } __Part VI — Scaling Out__

    ---

    **[11. Multi-GPU Systems, NCCL & Distributed Training](11-multi-gpu-nccl.md)**

    NVLink and NVSwitch topology, the DGX H100 node, ConnectX-7 and the
    inter-node bottleneck, the SuperPOD, rail networks and SHARP, InfiniBand
    leaf/spine/core, UVA and peer access, SLURM and PMIx, NCCL and the big six
    collectives, and data / tensor / pipeline / expert parallelism.

-   :material-rocket-launch:{ .lg .middle } __Part VII — Putting It to Work__

    ---

    **[12. FlashAttention-3 on the H100](12-flashattention-3.md)**

    Why attention is memory-bound and how fusing it crosses the ridge point;
    the online softmax recurrence worked by hand; then the three Hopper
    techniques — producer/consumer warp specialization, ping-pong scheduling,
    and 2-stage intra-warpgroup pipelining — that take it from 35% to 75% of
    peak. Causal masking and load balance, and the FP8 path: block
    quantization, the K-major transpose of V, and Hadamard incoherent
    processing.

    **[13. Profiling with Nsight Compute](13-nsight-compute.md)**

    Getting `ncu` to run, including the permission error that stops most
    first attempts; replay and what it does to your program; Speed of Light
    triage; the roofline chart and which ceiling actually applies on Hopper;
    sectors per request and bank conflicts; every warp stall reason mapped to
    the chapter that explains it; occupancy and why chasing it is wrong; the
    metric naming scheme; SASS and stall sampling; and four worked diagnoses
    from symptom to fix.

</div>

## Running this book locally

```bash
pip install -r requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000>.

To publish it to GitHub Pages, push to `main` — the workflow in
`.github/workflows/ci.yml` builds the site and deploys it for you. Edit the
`site_url` and `repo_url` lines at the top of `mkdocs.yml` to point at your own
repository first.

---

Built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
