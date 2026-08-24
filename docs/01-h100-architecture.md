# NVIDIA H100 GPU Architecture
## Lesson 1 Companion Guide — A Deep Dive into Hopper

> **Target audience:** GPU programmers who want to understand the H100 at the hardware level before writing CUDA/PTX kernels.
> **Prerequisites:** Basic understanding of parallel computing concepts (threads, memory hierarchy).

**What this covers:** Every architectural concept in Lesson 1 — the TMA unit, asynchronous execution, Tensor Cores, the memory hierarchy from HBM3 through L2/L1 to registers, the SM microarchitecture (quadrants, warp schedulers, dispatch units, SFUs, LSUs), NVLink 4.0, shared memory banking, swizzling, and how thread blocks are distributed across SMs. Each section includes ASCII diagrams, worked examples with concrete numbers, and practical implications for kernel design.

**How to read this chapter.** Most sections open with a boxed ASCII diagram or a
specification block. Those boxes are the reference material; the prose around
them tells you what to look at and why it matters. On a first pass, read the
prose and skim the boxes — you can come back for the exact numbers when a later
chapter needs them. Nothing here requires you to write code yet.

---

## Table of Contents

1. [The Big Picture — Why H100 Matters](#1--the-big-picture)
2. [Asynchronous Execution & The TMA](#2--asynchronous-execution--the-tma)
3. [4th Generation Tensor Cores](#3--4th-generation-tensor-cores)
4. [The Full GPU Die — Top-Level Organization](#4--the-full-gpu-die)
5. [HBM3 Memory System](#5--hbm3-memory-system)
6. [L2 Cache](#6--l2-cache)
7. [GPCs — Graphics Processing Clusters](#7--gpcs--graphics-processing-clusters)
8. [NVLink 4.0](#8--nvlink-40)
9. [The Streaming Multiprocessor (SM)](#9--the-streaming-multiprocessor-sm)
10. [SM Quadrants (SMSPs)](#10--sm-quadrants-smsps)
11. [Warp Schedulers & Dispatch](#11--warp-schedulers--dispatch)
12. [Execution Units Deep Dive](#12--execution-units-deep-dive)
13. [Unified Shared Memory + L1 Cache](#13--unified-shared-memory--l1-cache)
14. [Registers](#14--registers)
15. [Instruction Caches (L0 & L1)](#15--instruction-caches-l0--l1)
16. [Thread Block Distribution](#16--thread-block-distribution)
17. [PCIe 5.0 Host Interface](#17--pcie-50-host-interface)
18. [Putting It All Together — Data Flow Summary](#18--putting-it-all-together)
19. [Quick Reference Tables](#19--quick-reference-tables)

---

## 1 — The Big Picture

Before any code, you need a mental picture of what kind of machine this is. This
section gives you two things: the H100's raw numbers, and the one idea that
separates Hopper from every GPU before it. If you remember nothing else from
this chapter, remember §1.2.

### 1.1 H100 at a Glance

Here is the whole chip on a single card. Don't try to memorise it — skim it now
and come back whenever a later section mentions a number you want to place.
Three lines matter most for kernel work: **132 SMs** (how many independent
processors you are programming), **3.35 TB/s** of HBM3 bandwidth (how fast you
can feed them), and **228 KB of shared memory per SM** (how much data you can
stage right next to the compute).

```
NVIDIA H100 (Hopper Architecture)
══════════════════════════════════════════════════════════════════

Released:            September 2022
Process:             TSMC 4N (custom variant of N4)
Transistors:         80 billion
Form factors:        SXM5 (700W)  |  PCIe Gen5 (300W)

Compute:
  SMs:               132 (enabled)
  Tensor Cores:      528 (4 per SM)
  FP32 CUDA Cores:   16,896 (128 per SM)
  FP64 Units:        8,448 (64 per SM)
  INT32 Units:       8,448 (64 per SM)

Memory:
  HBM3:              80 GB @ 3.35 TB/s
  L2 Cache:          50 MB
  Shared Mem/SM:     228 KB (configurable)
  Registers/SM:      256 KB

Interconnect:
  NVLink 4.0:        900 GB/s (18 lanes)
  PCIe 5.0:          128 GB/s bidirectional

Key Innovation:      Fully asynchronous execution via TMA
```

### 1.2 The Generational Leap

The single most important thing to understand about the H100 is that it represents a **paradigm shift from synchronous to asynchronous execution**. Every previous GPU architecture required the SM's threads to babysit data movement — issuing loads, waiting for them, then computing. The H100 introduces hardware that does the data movement *in the background* while the SM computes on previously loaded data.

```
A100 (Ampere) — Synchronous Mental Model
═════════════════════════════════════════════════════════

  Thread 0 ──► Issue Load ──► WAIT... ──► Compute ──► Issue Load ──► WAIT...
                              ^^^^^^^^                               ^^^^^^^^
                              SM is idle                             SM is idle

  The thread cannot make progress until the load completes.
  Memory latency (hundreds of cycles) directly stalls compute.
  You hide latency by having many warps in flight, but there's a limit.


H100 (Hopper) — Asynchronous Mental Model
═════════════════════════════════════════════════════════

  Thread 0 ──► Tell TMA: "copy tile i+1" ──► Compute on tile i ──► Barrier ──►
               ^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^
               Returns immediately             TMA works in background

  The TMA unit handles the entire copy: stride calculation, bounds checking,
  address generation, and the actual data movement. The thread fires one
  instruction and immediately returns to useful compute.

  Result: Memory latency is HIDDEN under compute, not just tolerated.
```

This is why pipelined kernels (GEMMs, attention) can approach the theoretical roofline on the H100 — you fetch tile i+1 while computing tile i.

---

## 2 — Asynchronous Execution & The TMA

Section 1 claimed the H100 is asynchronous. This section makes that concrete.

The idea in one sentence: on older GPUs, threads had to fetch their own data and
then sit and wait for it; on the H100 a dedicated piece of hardware — the
**Tensor Memory Accelerator**, or TMA — does the fetching while those threads
keep computing.

We build up to it in four steps: what "asynchronous" means for a copy (§2.1),
what the TMA actually is (§2.2), a worked example where one TMA instruction
replaces roughly twenty thousand (§2.3), and the price you pay for all this —
bookkeeping (§2.4).

### 2.1 Synchronous vs Asynchronous Copies

A **synchronous copy** blocks the issuing thread (or warp) until the data arrives. The thread cannot execute any other instruction until the copy completes. Simple mental model: after the instruction, the data is ready.

An **asynchronous copy** only *starts* the transfer. The issuing thread can immediately continue doing other work — computing on previously loaded data, managing bookkeeping, etc. The thread later checks a barrier or fence to know when the copy has finished.

```
Synchronous Copy Timeline (per warp):
──────────────────────────────────────────────────────────────
  cycle 0        100       200       300       400       500
  │ load A ├─────wait─────┤ compute A│ load B ├──wait──┤ compute B│
  │        │   (stalled)  │          │        │(stall) │          │

  Utilization: ~50% (half the time waiting for memory)


Asynchronous Copy Timeline (per warp, pipelined):
──────────────────────────────────────────────────────────────
  cycle 0        100       200       300       400       500
  │ TMA: load B │ compute A │ TMA: load C │ compute B │ compute C│
  │ (returns    │           │ (returns    │           │          │
  │ immediately)│           │ immediately)│           │          │

  Utilization: ~95% (memory latency hidden under compute)
```

### 2.2 The TMA Unit

Each SM in the H100 has a dedicated **Tensor Memory Accelerator (TMA)** unit. This is a hardware block — it has its own logic, its own connection to the memory hierarchy, and its own understanding of multi-dimensional tensor layouts.

```
Before TMA (A100 and earlier):
═══════════════════════════════════════════════════════════

  To copy a 64×64 tile of FP16 from global → shared memory:

  Every thread in the warp must:
    1. Calculate its own source address (per-thread math)
    2. Calculate its own destination address
    3. Issue a load instruction
    4. Issue a store instruction
    5. Repeat in a loop for the entire tile region

  32 threads × multiple iterations = dozens of instructions
  Each thread burns registers for address math
  Coalescing depends on access pattern correctness


After TMA (H100):
═══════════════════════════════════════════════════════════

  A SINGLE thread issues ONE instruction with a descriptor:

    TMA Descriptor contains:
      ├── Base pointer (global memory start)
      ├── Tensor dimensions (e.g., [M, N])
      ├── Strides (bytes between rows/columns)
      ├── Box size (tile dimensions to copy)
      ├── Element type (FP16, FP8, etc.)
      └── Bounds (clamp out-of-range to zero)

    TMA hardware handles:
      ├── All address generation internally
      ├── Stride/offset calculations
      ├── Bounds checking (no if-statements needed!)
      ├── Data movement in the background
      └── Filling shared memory without SM intervention

  Result: 1 instruction replaces ~50+ instructions
  SM cores freed for actual compute
```

### 2.3 TMA Descriptor — Concrete Example

The previous section described the TMA in the abstract. Here is the same copy
written both ways, with real numbers, so you can watch the instruction count
collapse. Read the traditional block first and count what each thread has to do;
then read the TMA block.

```
Scenario: Copy a 64×64 tile of BF16 from a [4096, 4096] matrix
starting at row=128, col=256

Traditional approach (per thread, 32 threads in a warp):
──────────────────────────────────────────────────────────
  // Each thread handles 64*64/32 = 128 elements
  for (int i = threadIdx.x; i < 64*64; i += 32) {
      int row = i / 64;
      int col = i % 64;
      int src_offset = (128 + row) * 4096 + (256 + col);  // address math
      smem[row * 64 + col] = global[src_offset];            // load + store
  }
  __syncthreads();  // everyone must finish before compute

  Instructions per thread: ~128 loads + 128 stores + ~384 address math
  Total warp instructions: ~20,000+ instructions


TMA approach (single thread):
──────────────────────────────────────────────────────────
  // Thread 0 issues one instruction:
  tma_copy_2d(descriptor, /*box_coord=*/{128, 256}, smem_ptr, barrier);
  // All other threads: skip straight to compute

  // The descriptor was set up once at kernel launch:
  //   base_ptr    = &matrix[0][0]
  //   dims        = {4096, 4096}
  //   strides     = {4096 * 2, 2}   (2 bytes per BF16)
  //   box_size    = {64, 64}
  //   element_sz  = 2 bytes

  Instructions: 1 (literally one PTX instruction)
  The TMA hardware generates all 4096 memory transactions internally
```

### 2.4 The Bookkeeping Cost

The tradeoff for async execution is complexity. You need barriers to know when data is ready:

```
Async Pipeline Pattern (double-buffering):
══════════════════════════════════════════════════════════

  Shared memory has TWO tile slots: slot_A and slot_B

  Phase 0:
    TMA loads tile 0 → slot_A          ◄── async, returns immediately
    (nothing to compute yet, maybe setup work)

  Phase 1:
    Wait for slot_A to be ready         ◄── barrier wait
    TMA loads tile 1 → slot_B          ◄── async, returns immediately
    Compute on slot_A                   ◄── overlap!

  Phase 2:
    Wait for slot_B to be ready
    TMA loads tile 2 → slot_A          ◄── reuse slot_A (tile 0 consumed)
    Compute on slot_B                   ◄── overlap!

  ... and so on, ping-ponging between slots.

  Barriers used:
    mbarrier — hardware-accelerated async barrier
    arrive/wait pattern — producer signals, consumer waits
```

---

## 3 — 4th Generation Tensor Cores

Almost all of the H100's advertised performance lives in its tensor cores. A
CUDA core does one multiply-add at a time; a tensor core does an entire small
matrix multiply in one operation, which is where the 10–100× gap comes from.

This section covers what they do (§3.1), the new Hopper unit of work called a
*warp group* (§3.2), the trick of agreeing to skip zeros (§3.3), and the number
formats they accept (§3.4). You don't need to write tensor core code yet —
Chapters 6 and 7 do that. Here you only need to know what the hardware offers.

### 3.1 What Tensor Cores Do

Tensor Cores are specialized matrix-multiply-accumulate (MMA) units. While CUDA cores process one FMA (fused multiply-add) per cycle per core, a Tensor Core processes an entire small matrix multiply in one operation.

```
CUDA Core (FP32):
  1 core does: a × b + c → d     (1 FMA = 2 FLOPs per cycle)
  128 cores/SM × 2 FLOPs = 256 FLOPS/cycle/SM (FP32)

Tensor Core (FP16, 4th Gen):
  1 tensor core does: A[m×k] × B[k×n] + C[m×n] → D[m×n]
  Where m=16, k=16, n=16 for wgmma (with warp groups)
  4 tensor cores/SM working together via WGMMA:
    1024 FLOPS/cycle/SM (FP16)
    2048 FLOPS/cycle/SM (FP8)  ← 2× throughput at half precision

  FP8 peak for full GPU (132 SMs):
    2048 × 132 SMs = 270,336 FLOPS/cycle
    At ~1.8 GHz boost clock: 270,336 × 1.8 GHz ≈ 487 TFLOPS
    Spec sheet: 1,979 TFLOPS (with sparsity) / 989 TFLOPS (dense)
```

### 3.2 Warp Group MMA (WGMMA)

The H100 introduces **Warp Group** operations, where 4 warps (128 threads) cooperate on a single matrix multiply:

```
WGMMA — Warp Group Matrix Multiply Accumulate
══════════════════════════════════════════════════════════

  4 warps (128 threads) form a "warp group"
  They collectively compute: D = A × B + C

  Register layout requirements:
  ┌─────────────────────────────────────────────────────┐
  │  Matrix A:  Registers OR Shared Memory              │
  │  Matrix B:  MUST be in Shared Memory (never regs)   │
  │  Matrix C:  Distributed across 128 threads in regs  │
  │  Matrix D:  Same distribution as C (accumulated)    │
  └─────────────────────────────────────────────────────┘

  Each thread in the warp group holds a small piece of C/D.
  The tensor cores read A and B, multiply, and scatter-add
  the results into the correct threads' registers.

  Data packing:
    FP16: 2 values packed into 1 × 32-bit register
    FP8:  4 values packed into 1 × 32-bit register
    This is mandatory — WGMMA expects packed operands.
```

### 3.3 Structured Sparsity

Sparsity is a way of buying throughput by agreeing not to store some of your
numbers. The hardware supports exactly one pattern — "2:4" — and the payoff is
simple: the tensor core reads half as much data, so it runs about twice as fast.

```
Structured Sparsity (2:4 pattern):
══════════════════════════════════════════════════════════

  In every group of 4 consecutive elements, exactly 2 must be zero.

  Dense matrix row:      [ 1.5  0.0  3.2  0.0  0.0  2.1  0.0  4.7 ]
                           ^^^  ===  ^^^  ===  ===  ^^^  ===  ^^^
                           val  zero val  zero zero val  zero val

  Storage: Only the 4 non-zero values + a 4-bit index mask
    Values: [ 1.5, 3.2, 2.1, 4.7 ]   (50% compression)
    Mask:   [ 0,   2,   1,   3   ]    (which positions are non-zero)

  The tensor core decompresses on-the-fly during the MMA.
  Throughput doubles because you're reading half the data.

  FP8 with sparsity: 2048 × 2 = 4096 FLOPS/cycle/SM (theoretical)
  This is why the H100 spec says 1,979 TFLOPS (FP8, sparse).
```

### 3.4 Precision Comparison

Lower precision buys throughput: the fewer bits per number, the more numbers a
tensor core can chew through per cycle. This table is the menu. The pattern to
notice is that halving the bit width roughly doubles the FLOPs — which is the
entire reason FP8 exists.

```
Precision       Bits   Range              Tensor Core FLOPS/cycle   Use Case
─────────────────────────────────────────────────────────────────────────────
FP64            64     ±10^308            64/SM                     Scientific
TF32            32*    same as FP32       512/SM                    Training
FP32            32     ±3.4×10^38         (via CUDA cores only)     General
FP16            16     ±65504             1024/SM                   Training
BF16            16     ±3.4×10^38         1024/SM                   Training
FP8 (E4M3)     8      ±448               2048/SM                   Inference
FP8 (E5M2)     8      ±57344             2048/SM                   Gradients

* TF32 uses 32 bits storage but only 10-bit mantissa (≥10 bits)

The Transformer Engine dynamically selects FP8 E4M3 for forward
and E5M2 for backward, auto-scaling to prevent overflow.
```

---

## 4 — The Full GPU Die

Now we zoom out and look at the whole chip as a floor plan. Nearly everything
covered in the rest of this chapter is one of the boxes below, so it's worth a
minute here to see how the pieces sit relative to each other.

The rough shape: memory around the edges, 132 processors (SMs) in the middle
grouped into eight clusters, a large cache shared between them, and a hardware
scheduler deciding which processor runs what.

### 4.1 Top-Level Block Diagram

Read this from the outside in. Data either arrives at the bottom (over PCIe from
the CPU) or already lives at the top (in the HBM3 memory stacks). The SMs in the
middle are where your code actually runs, and the L2 cache between them is the
shared staging area that essentially all traffic passes through.

```
H100 SXM5 Die — Top Level
══════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────┐
  │                     NVLink Hub & Switches                        │
  │  (18 NVLink 4.0 lanes → 900 GB/s GPU-to-GPU bandwidth)         │
  ├────────┬────────┬────────┬────────┬────────┬────────┬───────────┤
  │ HBM3   │ HBM3   │ HBM3   │ HBM3   │ HBM3   │(HBM3) │           │
  │Stack 0 │Stack 1 │Stack 2 │Stack 3 │Stack 4 │ dis-  │           │
  │ 16GB   │ 16GB   │ 16GB   │ 16GB   │ 16GB   │abled  │           │
  ├────────┴────────┴────────┴────────┴────────┴───────┴───────────┤
  │              10 × 512-bit Memory Controllers                     │
  │         (DRAM scheduling, row buffering, address mapping)        │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
  │  │  GPC 0   │  │  GPC 1   │  │  GPC 2   │  │  GPC 3   │        │
  │  │ (18 SMs) │  │ (18 SMs) │  │ (18 SMs) │  │ (18 SMs) │        │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
  │  │  GPC 4   │  │  GPC 5   │  │  GPC 6   │  │  GPC 7   │        │
  │  │ (18 SMs) │  │ (18 SMs) │  │ (18 SMs) │  │ (18 SMs) │        │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
  │                                                                  │
  │          8 GPCs × 18 SMs/GPC = 144 SMs (132 enabled)            │
  │                                                                  │
  ├────────────────────────────────────────────────────────────────┤
  │             50 MB L2 Cache (25 MB per partition)                 │
  │           (128-byte lines, 32-byte sectors)                      │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │              GigaThread Engine                              │  │
  │  │  (CTA scheduling, cluster management, occupancy tracking)  │  │
  │  └────────────────────────────────────────────────────────────┘  │
  │                                                                  │
  │  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
  │  │  PCIe 5.0 Interface  │  │  Transformer Engine Controller   │  │
  │  │  (128 GB/s bidir)    │  │  (FP8 auto-scaling)              │  │
  │  └──────────────────────┘  └──────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────┘
```

### 4.2 The GigaThread Engine

The GigaThread Engine is the hardware scheduler that sits at the top of the GPU and manages the entire kernel launch lifecycle:

```
GigaThread Engine — Kernel Launch Flow
══════════════════════════════════════════════════════════

  CPU launches kernel via CUDA driver
          │
          ▼
  ┌─────────────────────────────────┐
  │       GigaThread Engine          │
  │                                  │
  │  1. Receives kernel descriptor   │
  │     (grid dims, block dims,      │
  │      shared mem, register count) │
  │                                  │
  │  2. Creates CTA (thread block)   │
  │     work queue                   │
  │                                  │
  │  3. For each unstarted CTA:      │
  │     ├── Check: does any SM have  │
  │     │   enough free resources?   │
  │     │   • Warp slots             │
  │     │   • Registers              │
  │     │   • Shared memory          │
  │     ├── YES → assign CTA to SM   │
  │     └── NO  → wait for SM to     │
  │              finish a CTA         │
  │                                  │
  │  4. Track CTA status:            │
  │     ├── NOT_STARTED              │
  │     ├── RUNNING                  │
  │     └── COMPLETED                │
  │                                  │
  │  [Hopper NEW] 5. Cluster-aware:  │
  │     Must schedule all CTAs in a   │
  │     cluster onto adjacent SMs     │
  │     within the same GPC           │
  └─────────────────────────────────┘
          │
          ▼
  CTA lands on an SM → SM begins executing warps
```

---

## 5 — HBM3 Memory System

HBM3 is the GPU's main memory — the 80 GB your tensors live in. It is
enormously wide (5,120 bits) and enormously fast in aggregate (3.35 TB/s), and
also, relative to the compute, enormously slow: roughly 500 cycles to answer a
request.

That tension — huge bandwidth, terrible latency — is the reason the rest of the
chip is shaped the way it is. Caches exist to shrink the number of requests that
have to travel this far; the TMA exists so you don't have to wait on the ones
that do.

### 5.1 Physical Organization

The part to take away is the bandwidth ladder at the bottom of the diagram:
about 33 TB/s right next to the SM, 12 TB/s at L2, 3.35 TB/s at HBM. Every step
further from the compute costs you roughly 3×.

```
HBM3 — High Bandwidth Memory, 3rd Generation
══════════════════════════════════════════════════════════

  Total capacity:    80 GB
  Bandwidth:         3.35 TB/s
  Stacks:            6 manufactured, 5 enabled (yield optimization)
  Per stack:         16 GB
  Bus width:         5120 bits total (1024 bits per stack)

  Memory controllers: 10 independent 512-bit controllers
  Each controller manages:
    ├── DRAM protocol timing (activate/precharge/read/write)
    ├── Request reordering (maximize row buffer hits)
    └── Address mapping (channels → banks → rows → cols)

  Data path from SM to HBM:
  ┌────┐    ┌────────┐    ┌──────────┐    ┌────────────┐    ┌──────┐
  │ SM │───►│L1D/    │───►│ L2 Cache │───►│ Memory     │───►│ HBM3 │
  │    │    │Coalesce│    │ Crossbar │    │ Controller │    │Stack │
  └────┘    └────────┘    └──────────┘    └────────────┘    └──────┘
             ~33 TB/s      ~12 TB/s        ~3.35 TB/s        ~3.35 TB/s
            (per SM)      (aggregate)      (aggregate)       (aggregate)

  The bandwidth drops at each stage further from the SM.
  This is WHY the memory hierarchy exists: each level filters traffic.
```

### 5.2 Why 5120-bit Bus Width Matters for TMA

A number like "5,120-bit bus" only becomes useful once you convert it into how
long your tile actually takes to arrive. That's what this does.

```
Bus width: 5120 bits = 640 bytes

TMA can move 128 bytes of data at once in a single transaction.
This fits cleanly into the bus:
  5120 bits / 8 = 640 bytes per bus cycle
  640 / 128 = 5 concurrent 128-byte transactions per cycle

For a 64×64 BF16 tile:
  64 × 64 × 2 bytes = 8192 bytes
  8192 / 128 = 64 transactions
  64 / 5 = ~13 bus cycles to transfer the entire tile

At ~1.8 GHz memory clock:
  13 cycles × ~0.56 ns/cycle ≈ 7.2 ns for one tile transfer
  This is fast enough to keep tensor cores fed.
```

---

## 6 — L2 Cache

The L2 is a 50 MB staging area shared by every SM, and essentially all traffic
to and from HBM passes through it.

Two facts about it drive real kernel performance: it is organised into 128-byte
lines made of four 32-byte **sectors**, and it charges you per *sector touched*,
not per useful byte. That single detail is what "coalescing" means, and it is
why two kernels that move the same amount of data can differ by 32× in speed.

### 6.1 Structure and Behavior

Four behaviours are described below. The first two — sectors, and the penalty
for scattered access — are the ones you'll actively think about while writing
kernels. The last two happen for you, for free.

```
L2 Cache — 50 MB Unified
══════════════════════════════════════════════════════════

  Total size:        50 MB
  Partitions:        2 × 25 MB
  Cache line:        128 bytes
  Sectors per line:  4 × 32 bytes
  Total lines:       50 MB / 128 B = 409,600 lines

  Key behaviors:

  1. SECTOR-BASED FILLING
     A cache line is 128 bytes, but consists of 4 × 32-byte sectors.
     A memory request can touch 1-4 sectors of a single line.
     The cache can fill sector-by-sector (no need to load all 128 bytes).

     ┌──────────┬──────────┬──────────┬──────────┐
     │ Sector 0 │ Sector 1 │ Sector 2 │ Sector 3 │  = 1 cache line
     │ 32 bytes │ 32 bytes │ 32 bytes │ 32 bytes │    (128 bytes)
     └──────────┴──────────┴──────────┴──────────┘

  2. UNCOALESCED ACCESS PENALTY
     A warp of 32 threads issues a memory instruction.
     If threads access scattered addresses:
       Best case:  1 sector  (all 32 threads in same 32B)
       Worst case: 32 sectors (every thread in a different sector)

     Uncoalesced access explodes the number of sectors requested,
     consuming L2 bandwidth and evicting useful cache lines.

  3. GPC-LOCAL PARTITIONING
     Each GPC is "closer" to one L2 partition.
     Accesses to the nearby partition have lower latency.
     The memory subsystem tries to map addresses so that
     an SM's working set hits its local L2 partition.

  4. WRITE COALESCING
     The L2 absorbs small, messy writes from SMs (via L1)
     and batches them into clean, large writes to HBM.
     This is critical for scatter/atomic operations.
```

### 6.2 Worked Example — L2 Sector Math

Same amount of data, two access patterns, a 32× difference in cost. Work through
both by hand — this is probably the most valuable arithmetic in the chapter.

```
Scenario: A warp of 32 threads each loads a float (4 bytes)
          from consecutive addresses starting at address 0x1000.

Thread  0: loads 4 bytes from 0x1000
Thread  1: loads 4 bytes from 0x1004
Thread  2: loads 4 bytes from 0x1008
...
Thread 31: loads 4 bytes from 0x107C

Total bytes: 32 × 4 = 128 bytes = exactly 1 cache line
Sectors touched: 4 (each sector = 32 bytes, 8 threads per sector)
L2 transactions: 1 cache line fetch (may fetch 1-4 sectors)

Result: PERFECTLY COALESCED — 1 L2 transaction for 32 threads.


Bad scenario: 32 threads each load a float with stride 128 bytes

Thread  0: 0x1000
Thread  1: 0x1080  (128 bytes later)
Thread  2: 0x1100
...
Thread 31: 0x1F80

Total bytes still 128, but spread across 32 different cache lines.
Sectors touched: 32 (one sector per thread, all different lines)
L2 transactions: 32 separate cache line fetches!

Result: 32× WORSE bandwidth usage for the same amount of data.
This is why coalescing matters so much.
```

---

## 7 — GPCs — Graphics Processing Clusters

The 132 SMs are not a flat pool. They are grouped into eight **Graphics
Processing Clusters** of 18 SMs each.

For most of GPU history this grouping was invisible to programmers. On Hopper it
stopped being invisible, because *thread block clusters* — groups of blocks that
can read each other's shared memory — only work within a single GPC. This is a
short section, but it is the reason cluster sizes cap out where they do.

### 7.1 GPC Internal Structure

One GPC, drawn out. The pieces to notice: the 18 SMs, their pairing into TPCs,
and the DSMEM interconnect that lets SMs read one another's shared memory
directly — without going out to L2 and back.

```
GPC — Graphics Processing Cluster
══════════════════════════════════════════════════════════

  One GPC contains:
    ├── 18 Streaming Multiprocessors (SMs)
    ├── 9 Texture Processing Clusters (TPCs), each holding 2 SMs
    ├── Connection to its local L2 cache partition
    └── Distributed Shared Memory (DSMEM) network

  ┌────────────────────────────────────────────────────────┐
  │                         GPC                             │
  │                                                         │
  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │
  │  │ TPC 0 │ │ TPC 1 │ │ TPC 2 │ │ TPC 3 │ │ TPC 4 │   │
  │  │SM0 SM1│ │SM2 SM3│ │SM4 SM5│ │SM6 SM7│ │SM8 SM9│   │
  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘   │
  │      │         │         │         │         │         │
  │      └─────────┴─────────┴─────────┴─────────┘         │
  │              DSMEM Interconnect (SM-to-SM)              │
  │                                                         │
  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐              │
  │  │ TPC 5 │ │ TPC 6 │ │ TPC 7 │ │ TPC 8 │              │
  │  │SM10 11│ │SM12 13│ │SM14 15│ │SM16 17│              │
  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘              │
  │      └─────────┴─────────┴─────────┘                    │
  │                                                         │
  ├─────────────────────────────────────────────────────────┤
  │              Connection to L2 Cache Partition            │
  │           (data flows: HBM → L2 → L1 via GPC)          │
  └─────────────────────────────────────────────────────────┘

  TPCs exist primarily to enable fast SM-to-SM communication
  for DSMEM. Two SMs within the same TPC can exchange data
  with minimal latency.

  CRITICAL: DSMEM only exists WITHIN a GPC.
  Cross-GPC communication must go through L2 or global memory.
```

### 7.2 Why GPCs Matter for Kernel Design

The practical consequence of everything above: how large can a cluster be, and
what does each size cost you in synchronisation?

```
Implication for Thread Block Clusters:
═══════════════════════════════════════

  A cluster of thread blocks must all run on SMs within the SAME GPC.
  (Because clusters need DSMEM, and DSMEM only exists within a GPC.)

  GPC has 18 SMs → maximum cluster size = 16 thread blocks
  (leaves ~2 SMs available for other work or yield failures)

  Practical cluster sizes:
    Size  2: 9 clusters per GPC, low sync overhead (87 cycles)
    Size  4: 4 clusters per GPC, moderate sync overhead
    Size  8: 2 clusters per GPC, good utilization
    Size 16: 1 cluster per GPC, ~2 SMs idle, 150-cycle sync cost
```

---

## 8 — NVLink 4.0

Everything so far has been inside a single GPU. NVLink is the wire *between*
GPUs — 900 GB/s of it, roughly seven times what PCIe offers.

You won't program NVLink directly in this chapter (Chapter 11 does that). What
matters here is its speed relative to PCIe and HBM, because that ratio is what
decides where to put your model once it stops fitting on one card.

### 8.1 Physical Specifications

The one piece of physics worth knowing is PAM4 signalling: by using four voltage
levels on the wire instead of two, each symbol carries two bits instead of one —
doubling the data rate without raising the clock.

```
NVLink 4.0 — GPU-to-GPU Interconnect
══════════════════════════════════════════════════════════

  Total bandwidth:     900 GB/s (bidirectional)
  Number of lanes:     18
  Sub-links:           9 (each = 2 lanes)
  Per sub-link:        100 GB/s bidirectional (50 GB/s per direction)

  Signaling:           PAM4 (Pulse Amplitude Modulation, 4 levels)
  ┌────────────────────────────────────────┐
  │  NRZ (old):  2 voltage levels → 1 bit per symbol     │
  │  PAM4 (new): 4 voltage levels → 2 bits per symbol    │
  │                                                        │
  │  Level 0: 0.0V  → bits "00"                           │
  │  Level 1: 0.3V  → bits "01"                           │
  │  Level 2: 0.6V  → bits "10"                           │
  │  Level 3: 0.9V  → bits "11"                           │
  │                                                        │
  │  Same wire frequency → 2× the data rate               │
  └────────────────────────────────────────┘

  Direct connections:  Up to 8 GPUs peer-to-peer
  NVSwitch fabric:     Up to 256 GPUs in a switched topology

  TMA integration:
    NVLink connects directly with the TMA unit.
    Within a DGX node, TMA can issue bulk copies to/from
    remote GPU memory without SM intervention.
```

### 8.2 Multi-GPU Topology

Eight GPUs, each able to reach every other at full speed. The bandwidth
comparison at the bottom is the part to remember.

```
DGX H100 Node (8 GPUs, fully connected via NVSwitch)
══════════════════════════════════════════════════════════

  ┌──────┐    NVLink     ┌──────┐
  │GPU 0 │◄────────────►│GPU 1 │
  │      │    900 GB/s    │      │
  └──┬───┘               └──┬───┘
     │ ╲                  ╱  │
     │   ╲    NVSwitch  ╱    │
     │     ╲          ╱      │
  ┌──┴───┐   ╲      ╱   ┌──┴───┐
  │GPU 2 │◄───╲──►╱───►│GPU 3 │
  └──┬───┘      ╳       └──┬───┘
     │        ╱   ╲         │
  ┌──┴───┐ ╱       ╲ ┌──┴───┐
  │GPU 4 │◄──────────►│GPU 5 │
  └──┬───┘            └──┬───┘
     │                    │
  ┌──┴───┐            ┌──┴───┐
  │GPU 6 │◄──────────►│GPU 7 │
  └──────┘            └──────┘

  Every GPU can talk to every other GPU at 900 GB/s.
  Total bisection bandwidth: 900 × 8 = 7.2 TB/s

  For comparison:
    PCIe 5.0: 128 GB/s (GPU ↔ CPU)
    NVLink:   900 GB/s (GPU ↔ GPU) → 7× faster than PCIe

  This is why multi-GPU training puts tensor/pipeline parallelism
  across NVLink, and data parallelism across network nodes.
```

---

## 9 — The Streaming Multiprocessor (SM)

This is the heart of the chapter. The SM is the unit that actually runs your
code: one thread block lands on one SM and stays there until it finishes.

The next several sections take the SM apart layer by layer — the four quadrants
inside it (§10), how each quadrant decides which warp to run (§11), the
arithmetic units they feed (§12), the shared memory they all share (§13), and
the registers each thread owns (§14). The diagram below is the map for all of
them.

### 9.1 SM Overview

The SM is the fundamental execution unit of the GPU. It receives thread blocks from the GigaThread Engine and executes them to completion.

```
H100 SM — Internal Architecture
══════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────┐
  │                    Streaming Multiprocessor                 │
  │                                                            │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │              L1 Instruction Cache (128 KB)           │  │
  │  │     (shared across all 4 quadrants)                  │  │
  │  └─────────────────────┬────────────────────────────────┘  │
  │                        │                                    │
  │  ┌──────────┬──────────┬──────────┬──────────┐            │
  │  │Quadrant 0│Quadrant 1│Quadrant 2│Quadrant 3│            │
  │  │ (SMSP 0) │ (SMSP 1) │ (SMSP 2) │ (SMSP 3) │            │
  │  │          │          │          │          │            │
  │  │ Warp     │ Warp     │ Warp     │ Warp     │            │
  │  │ Sched    │ Sched    │ Sched    │ Sched    │            │
  │  │          │          │          │          │            │
  │  │ L0 ICache│ L0 ICache│ L0 ICache│ L0 ICache│            │
  │  │          │          │          │          │            │
  │  │ 32 FP32  │ 32 FP32  │ 32 FP32  │ 32 FP32  │  = 128/SM │
  │  │ 16 FP64  │ 16 FP64  │ 16 FP64  │ 16 FP64  │  =  64/SM │
  │  │ 16 INT32 │ 16 INT32 │ 16 INT32 │ 16 INT32 │  =  64/SM │
  │  │ 1 Tensor │ 1 Tensor │ 1 Tensor │ 1 Tensor │  =   4/SM │
  │  │ 8 LSU    │ 8 LSU    │ 8 LSU    │ 8 LSU    │  =  32/SM │
  │  │ 4 SFU    │ 4 SFU    │ 4 SFU    │ 4 SFU    │  =  16/SM │
  │  │          │          │          │          │            │
  │  │ Dispatch │ Dispatch │ Dispatch │ Dispatch │            │
  │  │ Unit     │ Unit     │ Unit     │ Unit     │            │
  │  │          │          │          │          │            │
  │  │ Register │ Register │ Register │ Register │            │
  │  │  File    │  File    │  File    │  File    │            │
  │  │ (64 KB)  │ (64 KB)  │ (64 KB)  │ (64 KB)  │  = 256 KB │
  │  └──────────┴──────────┴──────────┴──────────┘            │
  │                        │                                    │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │        Unified Shared Memory / L1 Data Cache         │  │
  │  │            256 KB total (228 KB SMEM max)            │  │
  │  │         33 TB/s bandwidth, 32 banks × 4 bytes        │  │
  │  └──────────────────────────────────────────────────────┘  │
  │                        │                                    │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │                    TMA Unit                          │  │
  │  │   (async tensor copy, descriptor-based addressing)   │  │
  │  └──────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────┘
```

---

## 10 — SM Quadrants (SMSPs)

An SM is not one processor — it is four, and they schedule independently.

This matters for a reason you can act on immediately: if your block size isn't a
multiple of 128 threads, some of those four sit idle. §10.2 shows exactly how
that waste happens.

### 10.1 Quadrant Breakdown

Each SM is divided into 4 identical subdivisions. Each quadrant is an independently scheduling unit:

```
Single Quadrant (SMSP) — Detailed View
══════════════════════════════════════════════════════════

  ┌───────────────────────────────────────┐
  │            SMSP / Quadrant             │
  │                                        │
  │  ┌──────────────────────────────────┐  │
  │  │ Warp Scheduler                    │  │
  │  │  - Manages up to 16 warps         │  │
  │  │  - Selects 1 ready warp per cycle │  │
  │  │  - Maintains scoreboard           │  │
  │  └──────────────┬───────────────────┘  │
  │                  │                      │
  │  ┌──────────────▼───────────────────┐  │
  │  │ Dispatch Unit                     │  │
  │  │  - Routes instructions to pipes   │  │
  │  │  - Manages dispatch ports         │  │
  │  └──────────────┬───────────────────┘  │
  │                  │                      │
  │         ┌───────┼───────┐              │
  │         ▼       ▼       ▼              │
  │  ┌──────────┐ ┌────┐ ┌──────┐         │
  │  │ 32× FP32 │ │ 1× │ │ 16×  │         │
  │  │ CUDA     │ │Tens│ │ FP64 │         │
  │  │ Cores    │ │Core│ │ Units│         │
  │  └──────────┘ └────┘ └──────┘         │
  │                                        │
  │  ┌──────────┐ ┌─────────┐ ┌────────┐  │
  │  │ 16×INT32 │ │ 8× LSU  │ │ 4× SFU │  │
  │  │ Units    │ │ (Load/  │ │ (sin,  │  │
  │  │          │ │  Store)  │ │  cos,  │  │
  │  │          │ │          │ │  log)  │  │
  │  └──────────┘ └─────────┘ └────────┘  │
  │                                        │
  │  ┌──────────────────────────────────┐  │
  │  │ L0 Instruction Cache              │  │
  │  │ (private to this SMSP)            │  │
  │  └──────────────────────────────────┘  │
  │                                        │
  │  ┌──────────────────────────────────┐  │
  │  │ Register File: 64 KB             │  │
  │  │ (65,536 × 32-bit registers)      │  │
  │  │ Up to 128 reads/writes per cycle  │  │
  │  └──────────────────────────────────┘  │
  └───────────────────────────────────────┘

  Total per SM (4 quadrants):
    128 FP32 CUDA Cores
     64 FP64 Units
     64 INT32 Units
      4 Tensor Cores
     32 Load/Store Units
     16 Special Function Units
    256 KB Registers
      4 Warp Schedulers (16 warps each → 64 warps/SM max)
```

### 10.2 Why 4 Quadrants?

Here is the practical payoff of knowing the SM is four things rather than one.

```
Thread Block Distribution Example:
═══════════════════════════════════

  A thread block of 256 threads = 8 warps.
  Hardware distributes warps round-robin across quadrants:

  Quadrant 0: warp 0, warp 4   (warps 0-31, 128-159)
  Quadrant 1: warp 1, warp 5   (warps 32-63, 160-191)
  Quadrant 2: warp 2, warp 6   (warps 64-95, 192-223)
  Quadrant 3: warp 3, warp 7   (warps 96-127, 224-255)

  Each quadrant independently schedules its warps.
  All 4 quadrants execute in parallel every cycle.

  PRACTICAL RULE: Make block dimensions multiples of 128 threads
  (= 4 warps) to evenly distribute work across all 4 quadrants.

  Bad:  blockDim = 96 threads (3 warps)
    Q0: warp 0    Q1: warp 1    Q2: warp 2    Q3: (idle!)
    → 25% of the SM sits unused

  Good: blockDim = 128 threads (4 warps)
    Q0: warp 0    Q1: warp 1    Q2: warp 2    Q3: warp 3
    → All quadrants utilized
```

---

## 11 — Warp Schedulers & Dispatch

Threads on a GPU do not run individually. They run in locked groups of 32 called
**warps**, and the hardware's entire latency-hiding strategy is to keep many
warps resident and switch between them the moment one stalls.

This section explains what a warp is (§11.1), how the scheduler picks one each
cycle (§11.2), and a hardware limit most people never hear about — dispatch
ports (§11.3).

### 11.1 What is a Warp?

Start here if *warp* is a new word. A warp is simply the granularity at which
the hardware thinks: 32 threads that fetch one instruction and execute it
together, in lockstep.

```
Warp = 32 Threads
══════════════════════════════════════════════════════════

  Thread block of 256 threads:
    Warp 0: threads   0 -  31
    Warp 1: threads  32 -  63
    Warp 2: threads  64 -  95
    Warp 3: threads  96 - 127
    Warp 4: threads 128 - 159
    Warp 5: threads 160 - 191
    Warp 6: threads 192 - 223
    Warp 7: threads 224 - 255

  Properties:
    ├── Created together from a thread block
    ├── Share a warp scheduler's attention
    ├── Execute the SAME instruction (mostly — divergence aside)
    ├── Each thread has PRIVATE registers
    └── Memory operations coalesce across the 32 threads

  Why 32?
    ├── Matches 32 memory banks in shared memory
    ├── 32 threads × 4 bytes = 128 bytes = 1 cache line
    ├── Good balance between control overhead and parallelism
    └── Hardware sweet spot for SIMT execution
```

### 11.2 Warp Scheduler Operation

Five steps, every cycle, in every quadrant. The reason the whole machine is
built this way is the sentence at the bottom of the block.

```
Warp Scheduler — Per-Cycle Operation
══════════════════════════════════════════════════════════

  Each SMSP's warp scheduler manages up to 16 warps.
  Every cycle, it performs these steps:

  Step 1: SCAN all 16 warps
    ├── Check scoreboard: are source operands ready?
    ├── Check structural hazards: is the target pipe free?
    └── Check dependencies: any RAW/WAW hazards?

  Step 2: SELECT 1 ready warp
    ├── Priority heuristics (oldest, round-robin, etc.)
    └── The selected warp gets its next instruction issued

  Step 3: FETCH instruction
    ├── Send warp ID + program counter to L0 I-cache
    └── L0 hit → instruction available immediately
        L0 miss → fetch from L1 I-cache (higher latency)

  Step 4: DISPATCH
    ├── Send instruction to the Dispatch Unit
    ├── Dispatch Unit routes to correct functional unit
    └── Functional unit begins execution

  Step 5: UPDATE scoreboard
    ├── Mark destination register as "pending"
    ├── When execution completes → mark as "ready"
    └── This allows dependent warps to become eligible

  The key insight: by rapidly switching between 16 warps,
  the scheduler HIDES memory latency. While warp 0 waits
  for a load, warps 1-15 can be executing compute.
```

### 11.3 The Dispatch Unit and Dispatch Ports

This is the one part of the SM that no occupancy calculator will show you, and
it explains why "the SM has 128 FP32 cores and 64 INT32 units" does not mean a
single warp can use both in the same cycle.

```
Dispatch Ports — The Bottleneck You Don't See
══════════════════════════════════════════════════════════

  The dispatch unit has a limited number of PHYSICAL PORTS.
  Each port is a wire connecting to one or more functional units.

  ┌──────────┐
  │ Dispatch  │
  │   Unit    │
  │           │
  │  Port A ──┼──► FP32 CUDA Cores
  │           │         AND
  │           │    INT32 Units (SHARED port!)
  │           │
  │  Port B ──┼──► Tensor Core
  │           │
  │  Port C ──┼──► Load/Store Units
  │           │
  │  Port D ──┼──► Special Function Units
  │           │
  │  Port E ──┼──► FP64 Units
  └──────────┘

  CRITICAL RULE: Only ONE instruction per shared port per cycle.

  If Port A is shared between FP32 and INT32:
    Cycle N:   FP32 add instruction uses Port A
    Cycle N:   INT32 instruction CANNOT issue (port busy!)
    Cycle N+1: INT32 instruction can now issue

  This is why FP32 and INT32 can execute "simultaneously"
  (different warps) but the same warp can't issue both in
  the same cycle through the same port.

  The port is only needed for instruction LAUNCH, not the
  full execution duration. A tensor core MMA takes many cycles,
  but the port is freed after the first cycle.
```

---

## 12 — Execution Units Deep Dive

The five kinds of arithmetic hardware inside each quadrant, one at a time.

Most of this is reference material — skim it now and come back. The two
subsections that will actually change how you write code are §12.2 (integer work
runs in parallel with floating-point, so address arithmetic is effectively free)
and §12.5 (load/store units, and coalescing).

### 12.1 FP32 CUDA Cores

The default workhorse. Anything that isn't a matrix multiply ends up here.

```
FP32 CUDA Cores
══════════════════════════════════════════════════════════

  Per SM:        128 (32 per quadrant)
  Total (SXM5):  16,896 (132 SMs × 128)
  Operations:    Standard 32-bit floating point
                 (add, multiply, FMA, comparison, conversion)

  Throughput:    128 FP32 FMA/cycle/SM = 256 FLOPS/cycle/SM
                 (each FMA = 2 FLOPs: multiply + add)

  These are the "default" compute units — used for:
    ├── General-purpose math
    ├── Reduction operations
    ├── Activation functions (when SFU isn't appropriate)
    └── Any FP32 work that doesn't map to tensor cores
```

### 12.2 INT32 Units

The important claim here is the one in capitals: integer and floating-point work
proceed at the same time, on separate datapaths. This is why the address
arithmetic in a tiled loop costs you almost nothing.

```
INT32 Units
══════════════════════════════════════════════════════════

  Per SM:        64 (16 per quadrant)
  Total (SXM5):  8,448

  Primary uses:
    ├── Memory address calculations (pointer arithmetic)
    ├── Loop control (counters, comparisons)
    ├── Index math for tiling patterns
    └── General integer arithmetic

  KEY FEATURE: Can execute SIMULTANEOUSLY with FP32 cores.
  The hardware has separate datapaths for INT and FP.

  This means: while the FP32 pipeline computes on data,
  the INT32 pipeline can be computing the next set of
  addresses — zero stall for address generation.

  Typical pattern in a GEMM kernel:
    FP32: compute partial sum  (data path)
    INT32: calculate next tile address  (address path)
    Both execute in the SAME cycle (different warps or instructions).
```

### 12.3 FP64 Units

Double precision. On the H100 it runs at deliberately half the FP32 rate — this
chip was designed for machine learning, not for simulation.

```
FP64 Units
══════════════════════════════════════════════════════════

  Per SM:        64 (16 per quadrant)
  Total (SXM5):  8,448

  FP64 throughput = exactly HALF of FP32.
  This is by design — H100 prioritizes ML (FP16/FP8/BF16)
  over scientific computing (FP64).

  Used for:
    ├── High-precision scientific simulations
    ├── Financial computations requiring exact precision
    └── Numerical methods where FP32 isn't sufficient

  For comparison:
    A100:  FP64 = 1/2 FP32 throughput
    H100:  FP64 = 1/2 FP32 throughput  (same ratio)
    The absolute FP64 throughput is higher because the clock is higher.
```

### 12.4 Special Function Units (SFUs)

When your code calls `exp()` or `sqrt()`, it does not run a software routine.
These units approximate the answer in hardware, trading a little accuracy for a
great deal of speed.

```
Special Function Units (SFUs)
══════════════════════════════════════════════════════════

  Per SM:     16 (4 per quadrant)
  Throughput: 1 instruction per cycle per SFU → 16 ops/cycle/SM

  Functions handled:
    ├── sin(x), cos(x)           (trigonometric)
    ├── exp(x), exp2(x)          (exponential)
    ├── log(x), log2(x)          (logarithmic)
    ├── sqrt(x), rsqrt(x)        (roots)
    ├── rcp(x)                    (reciprocal: 1/x)
    └── Other transcendental functions

  Implementation:
    ├── Polynomial approximations (Chebyshev or minimax)
    ├── Lookup tables + interpolation
    └── Sacrifice some precision for massive throughput

  Precision: ~22 bits of mantissa accuracy (vs 23 for full FP32).
  For most ML workloads, this is more than sufficient.

  WARNING: If ALL 32 threads in a warp call an SFU function,
  and there are only 4 SFUs per quadrant, the operation takes:
    32 threads / 4 SFUs = 8 cycles per warp (for one quadrant)

  This is still much faster than computing sin() in software
  on the CUDA cores, which would take ~20+ instructions.
```

### 12.5 Load/Store Units (LSUs)

Read this one twice. Coalescing — how well the 32 addresses in a warp merge into
cache-line requests — is the single biggest lever on memory performance in most
kernels, and this is where it happens.

```
Load/Store Units (LSUs)
══════════════════════════════════════════════════════════

  Per SM:     32 (8 per quadrant)
  Connection: Direct to L1 cache / shared memory

  Operations:
    ├── ld  (load from memory → register)
    ├── st  (store from register → memory)
    ├── atom (atomic read-modify-write)
    └── red  (reduction operations)

  Coalescing (the most important concept for memory performance):
  ──────────────────────────────────────────────────────────────

  When a warp executes a load, 32 threads each provide an address.
  The LSU attempts to MERGE these 32 addresses into as few
  cache-line requests as possible.

  BEST CASE — perfectly coalesced:
    Thread 0: addr 0x1000
    Thread 1: addr 0x1004
    Thread 2: addr 0x1008
    ...
    Thread 31: addr 0x107C
    → ALL 32 addresses fit in 1 × 128-byte cache line
    → LSU issues 1 request to L1
    → 1 cache transaction serves all 32 threads

  WORST CASE — completely uncoalesced:
    Thread 0: addr 0x1000
    Thread 1: addr 0x2000
    Thread 2: addr 0x3000
    ...
    → Each thread hits a DIFFERENT cache line
    → LSU must issue 32 separate L1 requests
    → 32 transactions = 32× the bandwidth consumption

  When the L1 misses, the request cascades to L2 → HBM,
  multiplying the bandwidth waste at every level.
```

---

## 13 — Unified Shared Memory + L1 Cache

Shared memory is the scratchpad you manage by hand: 228 KB per SM, roughly 100×
quicker to reach than HBM, and the place your tiles live while the tensor cores
work on them.

This section covers how much of it you get (§13.1), how large to make your tiles
(§13.2), the one easy way to accidentally make it 32× slower (§13.3–§13.4), and
the fix for that (§13.5).

### 13.1 The Configurable Split

Shared memory and the L1 data cache are the same 256 KB of silicon, split
however you ask for it. More shared memory suits tiled algorithms that stage
data deliberately; more L1 suits irregular access patterns you can't predict
ahead of time.

```
Unified Shared Memory / L1 Data Cache
══════════════════════════════════════════════════════════

  Total size:     256 KB per SM
  Max shared mem: 228 KB (configurable)
  Bandwidth:      33 TB/s per SM
  Banks:          32 banks × 4 bytes (32 bits) wide
  Cache line:     128 bytes
  Sector:         32 bytes (cache fills sector-by-sector)

  The 256 KB is shared between:
    ├── Shared memory (programmer-managed, explicit)
    └── L1 data cache (hardware-managed, automatic)

  You choose the split:
  ┌────────────────────────────────────────────┐
  │          Total: 256 KB per SM               │
  │                                              │
  │  Option A:  228 KB shared  +  28 KB L1      │  ← GEMM kernels
  │  Option B:  164 KB shared  +  92 KB L1      │  ← balanced
  │  Option C:  100 KB shared  + 156 KB L1      │  ← cache-friendly
  │  Option D:   32 KB shared  + 224 KB L1      │  ← minimal shared
  └────────────────────────────────────────────┘

  TRADEOFF:
    More shared memory → better for tiled algorithms (GEMM, attention)
                         but less L1 cache for spill/register pressure
    More L1 cache → better for irregular access patterns
                    but less space for explicit data staging

  Per-thread-block limit: 227 KB (1 KB reserved by CUDA runtime)
  Static __shared__ arrays: limited to 48 KB (compatibility)
  Beyond 48 KB: must use dynamic shared memory (extern __shared__)
```

### 13.2 Optimal Tile Sizing

Bigger tiles are not automatically better. Here is how to think about the
tradeoff in both directions.

```
Practical Tile Size Guidelines for GEMMs:
══════════════════════════════════════════════════════════

  The sweet spot is 64-128 KB per thread block.

  Why not use all 228 KB?
    ├── Larger tiles → more data to load before computing
    ├── First iteration has nothing to overlap with
    ├── Higher latency to fill the tile
    └── Fewer thread blocks fit per SM (hurts occupancy)

  Why not tiny tiles (e.g., 8 KB)?
    ├── Tile overhead (setup, sync) amortized over less compute
    ├── Lower compute-to-memory ratio
    └── More tiles needed → more kernel launch overhead

  Example for a BF16 GEMM:
    Tile = 128 × 128 × 2 bytes = 32 KB per matrix
    A tile + B tile = 64 KB ← fits well in shared memory
    Leaves room for double-buffering: 64 KB × 2 = 128 KB
    Still within the 228 KB limit with room for L1 cache
```

### 13.3 Bank Conflicts

Shared memory is fast because it is split into 32 independent **banks** that can
all be read in the same cycle. It stops being fast the instant two threads want
different addresses that happen to live in the same bank.

```
Shared Memory Banks
══════════════════════════════════════════════════════════

  32 banks, each 4 bytes wide.
  Banks are interleaved across consecutive 4-byte words:

  Address  0x00 - 0x03 → Bank  0
  Address  0x04 - 0x07 → Bank  1
  Address  0x08 - 0x0B → Bank  2
  ...
  Address  0x7C - 0x7F → Bank 31
  Address  0x80 - 0x83 → Bank  0  ← wraps around!
  Address  0x84 - 0x87 → Bank  1

  Bank index formula: bank = (byte_address / 4) % 32

  NO CONFLICT:
    32 threads access 32 different banks → all served in 1 cycle
    OR multiple threads read the SAME address → broadcast in 1 cycle

  CONFLICT:
    2+ threads access DIFFERENT addresses in the SAME bank
    → serialized: N-way conflict takes N cycles
```

### 13.4 Worked Example — Bank Conflict

The worst case, worked out. A perfectly reasonable-looking access — read one
column of a matrix — collapses to a thirty-second of full speed.

```
Scenario: 32 threads load column 0 of a 64×64 BF16 matrix
stored row-major in shared memory.

Row layout: 64 elements × 2 bytes = 128 bytes per row

Thread 0 loads: row 0, col 0 → address 0x0000 → bank (0/4)%32 = 0
Thread 1 loads: row 1, col 0 → address 0x0080 → bank (128/4)%32 = 0
Thread 2 loads: row 2, col 0 → address 0x0100 → bank (256/4)%32 = 0
...
Thread 31 loads: row 31, col 0 → address 0x0F80 → bank = 0

ALL 32 THREADS HIT BANK 0!
This is a 32-way bank conflict → takes 32 cycles instead of 1.
That's a 32× performance loss on this access!

Why?
  Row size = 128 bytes = 32 × 4-byte words = exactly 32 banks
  So column 0 of every row maps to the same bank.
  stride (128 bytes) is a perfect multiple of bank width (128 bytes).

This is EXACTLY why swizzling exists.
```

### 13.5 Swizzling — The Fix for Bank Conflicts

The fix is to scramble where each row starts, so that a column no longer lands
in a single bank. That is swizzling — and the good news at the end is that the
TMA can apply it for you on the way in.

```
Swizzling — Rearranging Data Layout to Avoid Conflicts
══════════════════════════════════════════════════════════

  Problem: Matrix[row][0] always maps to bank 0 for all rows.
  Solution: XOR the row index into the bank calculation.

  Without swizzling (row-major, 64×64 BF16):
  ┌─────────────────────────────────────────────┐
  │ Row 0: bank 0,1,2,...31,0,1,2,...31         │
  │ Row 1: bank 0,1,2,...31,0,1,2,...31         │  ← column 0
  │ Row 2: bank 0,1,2,...31,0,1,2,...31         │     always bank 0!
  │ ...                                          │
  └─────────────────────────────────────────────┘

  With swizzling (XOR row with column group):
  ┌─────────────────────────────────────────────┐
  │ Row 0: bank 0,1,2,...31,0,1,...              │
  │ Row 1: bank 1,2,3,...0,1,2,...               │  ← column 0 is
  │ Row 2: bank 2,3,4,...1,2,3,...               │     bank 0,1,2,3...
  │ Row 3: bank 3,4,5,...2,3,4,...               │     all different!
  │ ...                                          │
  └─────────────────────────────────────────────┘

  Swizzle formula (simplified):
    swizzled_bank = bank XOR (row % 32)

  Column 0 access after swizzling:
    Thread 0 (row 0): bank = 0 XOR 0 = 0
    Thread 1 (row 1): bank = 0 XOR 1 = 1
    Thread 2 (row 2): bank = 0 XOR 2 = 2
    Thread 3 (row 3): bank = 0 XOR 3 = 3
    ...
    Thread 31 (row 31): bank = 0 XOR 31 = 31

    ALL 32 threads hit DIFFERENT banks → conflict-free!

  The hardware (TMA) can apply swizzling automatically
  when copying into shared memory, so you don't need
  to do the XOR manually in your kernel code.
```

---

## 14 — Registers

Registers are the fastest storage on the chip and the scarcest: 256 KB per SM,
divided among every thread resident there.

The tension running through this section is that using more registers per thread
makes each thread faster, but lets fewer threads run at once — and past a
certain point that trade goes badly.

### 14.1 Register File Specifications

The specifications, plus the packing rules. Note that FP16 stores two values per
register and FP8 stores four — which is why tensor core operands have to be
packed before you hand them over.

```
Register File
══════════════════════════════════════════════════════════

  Per SM:        256 KB (65,536 × 32-bit registers)
  Per quadrant:  64 KB  (16,384 × 32-bit registers)
  Bandwidth:     Up to 128 reads/writes per cycle per SM
  Latency:       1 cycle (lowest of any storage)
  Energy:        30-50× more efficient than shared memory
                 1000×+ more efficient than HBM3

  Every thread gets a PRIVATE set of registers.
  Register allocation is determined at COMPILE TIME.

  Base unit: 32-bit slots
    ├── FP32:    1 register per value
    ├── FP64:    2 registers per value (register pair)
    ├── FP16:    2 values packed into 1 register
    ├── BF16:    2 values packed into 1 register
    ├── FP8:     4 values packed into 1 register
    ├── INT32:   1 register per value
    └── Pointers: 2 registers (64-bit address)
```

### 14.2 Occupancy Limiting

The tradeoff in numbers. Watch for the cliff: because the compiler allocates
registers in fixed-size chunks, two extra registers per thread can cost you an
entire warp's worth of occupancy.

```
Register Pressure — The Silent Performance Killer
══════════════════════════════════════════════════════════

  Per quadrant: 16,384 registers
  If your kernel uses R registers per thread:

  Max threads per quadrant = floor(16,384 / R)
    But also capped by other limits (warps, shared mem, etc.)

  Example calculations:
  ─────────────────────────────────────────────────
  Regs/thread    Max threads/quadrant    Max warps/quadrant
  ─────────────────────────────────────────────────
      32          512                     16 (max)
      64          256                      8
      96          170                      5
     128          128                      4
     160          102                      3
     192           85                      2
     255           64                      2
  ─────────────────────────────────────────────────

  CRITICAL CLIFF EFFECT:
    63 registers/thread → floor(16384/63) = 260 → 8 warps
    65 registers/thread → floor(16384/65) = 252 → 7 warps

    That's a 12.5% drop in occupancy from adding 2 registers!

    Register allocation has internal GRANULARITY (typically 8 regs).
    The compiler rounds UP to the next granularity boundary.
    A small change in code can push you over a boundary.
```

### 14.3 Register Spilling

What happens when you lose that fight, and how to notice that you have.

```
What Happens When You Run Out of Registers
══════════════════════════════════════════════════════════

  When the compiler needs more registers than the hardware has,
  it "spills" values to slower memory:

  Priority (best to worst):
    1. Registers          ← 1 cycle, lowest energy
    2. Shared memory      ← ~20 cycles (CUDA 13.0+ spill target)
    3. Local memory       ← ~200-400 cycles (lives in L2/HBM!)

  CUDA 13.0 improvement:
    Before: spill → local memory (through L2/HBM, very slow)
    After:  spill → shared memory first (much faster)
            Only falls back to local memory if shared mem is full

  How to detect spilling:
    nvcc --ptxas-options=-v  →  shows "spill loads" and "spill stores"
    If you see any, your kernel is likely register-pressure limited.

  How to reduce register usage:
    ├── __launch_bounds__(maxThreads, minBlocks)
    ├── Recompute values instead of storing them
    ├── Use smaller data types (FP16 → half the registers)
    └── Restructure code to reduce live variables
```

### 14.4 WGMMA Register Layout

A forward reference to Chapters 6 and 7. Tensor cores are particular about where
their operands live, and the rules are not symmetric between A, B, and C.

```
WGMMA Register Layout Requirements
══════════════════════════════════════════════════════════

  For Tensor Core operations (WGMMA):

  Matrix A:
    ├── Can be in shared memory OR registers
    ├── If in registers: must follow specific thread-to-element mapping
    └── FP16/BF16: packed 2 per 32-bit register

  Matrix B:
    ├── MUST be in shared memory (SMEM)
    ├── NEVER in registers
    └── This is a hard architectural constraint

  Matrix C (accumulator):
    ├── Distributed across 4 warps (128 threads)
    ├── Each thread holds a small fragment of the output
    ├── Stays in registers throughout the MMA
    └── FP32 accumulation is standard (higher precision)

  Packing requirements:
    FP16:  2 values per 32-bit register → register.xy = {val0, val1}
    BF16:  2 values per 32-bit register → same packing
    FP8:   4 values per 32-bit register → register.xyzw = {v0,v1,v2,v3}

    WGMMA instructions EXPECT packed data. If your data
    isn't packed correctly, the instruction produces garbage.
```

---

## 15 — Instruction Caches (L0 & L1)

Instructions have to be fetched too, and the same caching logic applies to them
as to data. There are two levels.

The only practical takeaway is at the end of §15.1: unrolling a loop too
aggressively can push your inner loop out of the tiny L0 cache and cost you more
than the unrolling saved.

### 15.1 L0 Instruction Cache

A very small buffer, private to each quadrant, whose entire job is to hold your
inner loop so the scheduler never waits on an instruction fetch.

```
L0 Instruction Cache — Per-Quadrant Micro-Buffer
══════════════════════════════════════════════════════════

  Purpose:    Feed the warp scheduler with instructions at wire speed
  Scope:      Private to each SMSP (quadrant 0 cannot use quadrant 1's L0)
  Size:       Very small (a few tight loops fit, not much more)
  Hit rate:   Extremely high for inner loops (GEMM, convolution)

  Without L0:
    Every cycle, the scheduler would have to fetch from L1.
    L1 is shared across all 4 quadrants → contention.
    L1 has higher latency → pipeline bubbles.

  With L0:
    Tight loops are captured in L0 after first execution.
    Subsequent iterations: instruction fetch = 1 cycle.
    The scheduler issues at full speed without waiting.

  What BREAKS the L0:
    ├── Aggressive loop unrolling (code doesn't fit in L0)
    ├── Heavy function inlining (expanded code too large)
    ├── Very branchy code (cache thrashing)
    └── Solution: balance unrolling vs. L0 cache pressure
```

### 15.2 L1 Instruction Cache

The larger backstop behind L0, shared by all four quadrants.

```
L1 Instruction Cache — Shared Across SM
══════════════════════════════════════════════════════════

  Size:     128 KB per SM
  Purpose:  Buffer SASS instructions for all 4 quadrants

  Without L1 I-Cache:
    Every instruction would come from L2 cache or HBM3.
    L2 access: ~30-50 cycles
    HBM3 access: ~300-500 cycles
    The pipeline would stall constantly.

  With L1 I-Cache:
    Kernel code is loaded once into L1.
    All 4 quadrants fetch from the same L1.
    Common loops served at ~5-10 cycle latency.

  The L1 I-Cache also handles:
    ├── Branch target buffering (jumps, loops)
    ├── Function call/return target caching
    └── Prefetching sequential instruction streams

  GEMM inner loop example:
    The inner loop of a tiled GEMM is typically ~50-200 instructions.
    This easily fits in the L1 I-Cache (128 KB).
    After the first iteration, all instruction fetches hit L1.
```

---

## 16 — Thread Block Distribution

This section closes the loop between the code you write — `kernel<<<grid,
block>>>` — and the hardware you've just been reading about.

Two questions get answered: how one block is placed on an SM and split into
warps (§16.1), and how many blocks can share an SM at once (§16.2). That second
number is what people mean when they say *occupancy*.

### 16.1 How CTAs Land on SMs

Follow one thread block from the scheduler through to execution. The step that
surprises most people is step 4: the warps of a single block are spread across
all four quadrants rather than kept together — which is why `__syncthreads()`
needs real barrier hardware behind it.

```
Thread Block (CTA) → SM Assignment
══════════════════════════════════════════════════════════

  1. GigaThread Engine picks an SM with enough resources:
     ├── Free warp slots (to hold the block's warps)
     ├── Free registers (block_size × regs_per_thread)
     ├── Free shared memory (block's shared mem allocation)
     └── [Hopper] Cluster constraint (adjacent SMs in same GPC)

  2. The SM receives the CTA and begins execution.

  3. The hardware divides the CTA into warps:
     ├── Warp 0: threads   0 - 31
     ├── Warp 1: threads  32 - 63
     ├── Warp 2: threads  64 - 95
     └── ...

  4. Warps are assigned to SMSPs in round-robin:
     SMSP 0: warp 0, warp 4, warp 8, ...
     SMSP 1: warp 1, warp 5, warp 9, ...
     SMSP 2: warp 2, warp 6, warp 10, ...
     SMSP 3: warp 3, warp 7, warp 11, ...

  KEY: The hardware does NOT keep a block in a single SMSP
  (unless the block is tiny, e.g., 32 threads = 1 warp).
  The warps are SPREAD across all 4 SMSPs.

  IMPORTANT FOR __syncthreads():
     When you call __syncthreads(), warps in DIFFERENT SMSPs
     must synchronize. This is handled by barrier logic in the
     shared memory block (which all SMSPs share).
     Other warps are put to sleep until every warp in the
     thread block has reached the barrier.
```

### 16.2 Multiple CTAs Per SM

Occupancy in one worked example. Whichever resource runs out first — warp slots,
registers, or shared memory — is the one that sets your limit.

```
Multiple CTAs Running Concurrently on One SM
══════════════════════════════════════════════════════════

  An SM can run MULTIPLE thread blocks simultaneously,
  as long as total resources don't exceed limits.

  Example:
    SM capacity: 64 warps, 256 KB registers, 228 KB shared mem

    Block A: 128 threads (4 warps), 64 regs/thread, 32 KB shared
      Resources: 4 warps, 32 KB regs, 32 KB shared

    How many Block A's fit?
      Warps:  64 / 4 = 16 blocks
      Regs:   256 KB / 32 KB = 8 blocks
      Shared: 228 KB / 32 KB = 7 blocks
      Answer: min(16, 8, 7) = 7 blocks per SM

    With 7 blocks: 28 warps active, 224 KB regs, 224 KB shared
    That's 28/64 = 43.75% warp occupancy.

  Higher occupancy → more warps to hide latency → usually better.
  But diminishing returns after ~50-60% for compute-bound kernels.
```

---

## 17 — PCIe 5.0 Host Interface

The last link, and by a wide margin the slowest: the connection back to the CPU.
There is not much to design here, only one rule to internalise, and the
bandwidth comparison below makes it obvious.

```
PCIe 5.0 — CPU ↔ GPU Connection
══════════════════════════════════════════════════════════

  Bandwidth:     128 GB/s bidirectional (64 GB/s per direction)
  Generation:    PCIe Gen 5, x16 lanes

  Primary roles:
    1. Data highway: Transfer data between CPU memory and GPU memory
    2. Kernel launch: CPU sends kernel descriptors to GPU
    3. GPUDirect RDMA: Network cards (NICs) can read/write GPU
       memory directly without CPU involvement

  Bandwidth comparison:
    ├── HBM3:      3,350 GB/s (internal GPU memory bandwidth)
    ├── NVLink:      900 GB/s (GPU-to-GPU)
    ├── PCIe 5.0:    128 GB/s (CPU-to-GPU)
    └── Ethernet:     50 GB/s (400GbE network)

  PCIe is the SLOWEST link in the system.
  This is why:
    ├── Model weights should be loaded once and stay on GPU
    ├── Minimize CPU↔GPU data transfers during inference
    ├── GPUDirect RDMA bypasses CPU for network → GPU transfers
    └── NVLink (not PCIe) is used for multi-GPU communication
```

---

## 18 — Putting It All Together

Everything in this chapter, assembled into the path a single GEMM tile actually
takes: from host memory into HBM, up through L2 into shared memory, into
registers, through the tensor cores, and back out again.

If the individual sections felt disconnected, this is where they join up.

### 18.1 Complete Data Flow

Read the five steps in order, then read the note at the bottom — the entire point
is that steps 2 and 4 are happening at the same time, for different tiles.

```
Data Flow Through the H100 — End to End
══════════════════════════════════════════════════════════════════

  A single GEMM tile operation:

  Step 1: TMA DESCRIPTOR SETUP (once, at kernel launch)
  ┌───────────────────────────────────────────┐
  │ CPU creates TMA descriptor:               │
  │   base_ptr, dims, strides, box_size       │
  │ Transfers to GPU via PCIe 5.0             │
  └───────────────────────────────────────────┘
                    │
                    ▼

  Step 2: GLOBAL → SHARED MEMORY (TMA, async)
  ┌───────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
  │  HBM3     │───►│ Memory    │───►│ L2 Cache │───►│ Shared   │
  │  Stack    │    │Controller │    │ (50 MB)  │    │ Memory   │
  │  (80 GB)  │    │ (reorder) │    │          │    │ (228 KB) │
  └───────────┘    └───────────┘    └──────────┘    └──────────┘
  3.35 TB/s         scheduling      ~12 TB/s agg    33 TB/s/SM
                    + row hits

  Step 3: SHARED MEMORY → REGISTERS (warp loads)
  ┌──────────┐                    ┌──────────┐
  │ Shared   │───────────────────►│ Register │
  │ Memory   │    (bank access)   │   File   │
  │ (228 KB) │    1 cycle if      │ (256 KB) │
  └──────────┘    no conflicts    └──────────┘
  33 TB/s/SM                      highest BW

  Step 4: COMPUTE (Tensor Cores / CUDA Cores)
  ┌──────────┐                    ┌──────────┐
  │ Register │───────────────────►│ Tensor   │
  │   File   │    operands        │  Cores   │──► results back
  │          │                    │ (WGMMA)  │    to registers
  └──────────┘                    └──────────┘
              1024-2048 FLOPS/cycle/SM

  Step 5: REGISTERS → SHARED MEMORY → GLOBAL (write back)
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
  │ Register │───►│ Shared   │───►│ L2 Cache │───►│  HBM3     │
  │   File   │    │ Memory   │    │ (coalesce│    │  Stack    │
  └──────────┘    └──────────┘    │  writes) │    └───────────┘
                                  └──────────┘

  The entire pipeline is OVERLAPPED:
    While Step 4 computes tile i,
    Step 2 loads tile i+1 via TMA (async).
    This is the fundamental performance trick of the H100.
```

### 18.2 Memory Hierarchy Summary

The hierarchy on one card. The right-hand columns are the ones worth
internalising: each level down costs roughly an order of magnitude more latency
and 10× more energy per access.

```
H100 Memory Hierarchy — Bandwidth vs Capacity
══════════════════════════════════════════════════════════

  Level          Size        Bandwidth       Latency    Energy
  ──────────────────────────────────────────────────────────────
  Registers     256 KB/SM    ~128 R/W/cycle   1 cycle   1×
  Shared Mem    228 KB/SM    33 TB/s/SM      ~20 cyc    30-50×
  L1 Cache      28-224 KB    (part of shared) ~30 cyc   30-50×
  L2 Cache      50 MB        ~12 TB/s agg    ~200 cyc   200×
  HBM3          80 GB        3.35 TB/s       ~500 cyc   1000×+
  ──────────────────────────────────────────────────────────────

  The ratio:
    HBM3 → L2:    ~3.5×  bandwidth amplification
    L2   → SMEM:  ~2.8×  per SM (33 TB/s vs 12 TB/s / 132 SMs)
    SMEM → Regs:  ~4×    effective (no bank conflicts)

  Each level filters traffic for the next:
    If L1 hits 90%, only 10% of requests reach L2.
    If L2 hits 80%, only 20% of L2 requests reach HBM.
    Total HBM traffic = 10% × 20% = 2% of original requests.

  This cascading filtering is what makes the hierarchy work.
```

---

## 19 — Quick Reference Tables

Nothing new here — these are the numbers from the chapter collected in one place
so you can come back to them without re-reading. The last block, the three laws,
is this entire chapter compressed to nine lines.

### Execution Unit Counts (SXM5, 132 SMs)

```
Unit                Per Quadrant    Per SM     Total GPU
──────────────────────────────────────────────────────────
FP32 CUDA Cores         32          128        16,896
FP64 Units              16           64         8,448
INT32 Units             16           64         8,448
Tensor Cores             1            4           528
Load/Store Units         8           32         4,224
Special Function Units   4           16         2,112
Warp Schedulers          1            4           528
Max Warps               16           64         8,448
```

### Memory Specifications

```
Component            Size        Bandwidth      Line/Sector
──────────────────────────────────────────────────────────────
HBM3                 80 GB       3.35 TB/s      —
L2 Cache             50 MB       ~12 TB/s       128B / 32B
Shared Mem (max)    228 KB/SM    33 TB/s/SM     32 banks × 4B
L1 Data Cache       28-224 KB    (part of shared) 128B / 32B
Register File       256 KB/SM    128 R/W/cycle   32-bit slots
L1 I-Cache          128 KB/SM    —               —
L0 I-Cache          tiny/SMSP    1 insn/cycle    —
```

### Key Formulas

```
Occupancy Limiters:
  Max warps from registers: floor(65536 / regs_per_thread) / 32
  Max warps from shared: floor(228 KB / shared_per_block) × warps_per_block
  Max warps from hardware: 64 per SM (16 per quadrant)
  Actual occupancy = min(all three) / 64

Bank Conflict:
  bank = (byte_address / 4) % 32
  Conflict: 2+ threads access different addresses in same bank
  No conflict: all threads access different banks OR same address

Swizzle:
  swizzled_bank = bank XOR (row_index % 32)

Tensor Core throughput:
  FP16: 1024 FLOPS/cycle/SM (dense)
  FP8:  2048 FLOPS/cycle/SM (dense)
  Sparse: 2× dense throughput
```

### The Three Laws of H100 Performance

```
1. ASYNC EVERYTHING
   Use TMA for all bulk data movement.
   Overlap loads with compute via double-buffering.
   Never block an SM waiting for memory.

2. FEED THE TENSOR CORES
   Tensor cores deliver 10-100× the throughput of CUDA cores.
   If your kernel isn't using tensor cores, you're leaving
   90%+ of the chip's capability on the table.

3. RESPECT THE HIERARCHY
   Registers > Shared Mem > L2 > HBM3
   Every access to a lower level is 10-1000× more expensive.
   Design your data layout to maximize hits at each level.
```

---

> **What's Next:** Lesson 2 covers Thread Block Clusters, Distributed Shared Memory, PTX inline assembly, data types, pointers, and state spaces — the programming model that unlocks the hardware described here.