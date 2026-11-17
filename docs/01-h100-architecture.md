# NVIDIA H100 GPU Architecture
## Lesson 1 Companion Guide — A Deep Dive into Hopper

[![CHAPTER 01: Inside the Hopper machine](images/chapter-openers/01-architecture.svg)](images/chapter-openers/01-architecture.svg)

*Architecture introduction years: [Volta (2017)](https://nvidianews.nvidia.com/news/nvidia-launches-revolutionary-volta-gpu-platform-fueling-next-era-of-ai-and-high-performance-computing), [Ampere (2020)](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/), [Hopper (2022)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [Blackwell (2024)](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/). Package drawings are conceptual, not die photographs or scale comparisons.*


> **Validation — November 17, 2026.** Architecture reference. SXM product specifications and models are distinguished from the PCIe execution host.
> **Follow-up:** the [final two-H100 audit](final-audit.md) records a fresh run of the complete programs. The original dated transcripts below remain available for provenance.
> **Execution system:** 1× H100 PCIe, 114 SMs, 80 GB class; Ubuntu 24.04.5 LTS; driver 580.105.08; CUDA Toolkit 12.8.93; `sm_90a`, C++17, `-O2`.
> [Full system information](validation/2026-11-17/system-information.txt) · [CUDA result manifest](validation/2026-11-17/results.json). Each tested source and captured output is linked beside its example. Diagrams open as original SVGs for zooming.


> **Target audience:** GPU programmers who want to understand the H100 at the hardware level before writing CUDA/PTX kernels.
> **Prerequisites:** Basic understanding of parallel computing concepts (threads, memory hierarchy).


<!-- learning:start -->
!!! tip "Your goal in this chapter"

    Separate product specifications from a throughput model; identify whether a number is per SM, per GPU, or bidirectional.

    No prior Hopper knowledge. Read the [installation guide](getting-started.md) if you want to check the execution host.
<!-- learning:end -->

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
**SKU scope:** the specification blocks below describe H100 SXM unless stated otherwise. The November 17, 2026 execution records use a single H100 PCIe; do not apply SXM SM counts, HBM3 bandwidth, NVLink topology, or peak throughput to that machine.

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
  NVLink 4.0:        900 GB/s (18 links)
  PCIe 5.0:          128 GB/s bidirectional

Key Innovation:      Fully asynchronous execution via TMA
```

### 1.2 The Generational Leap

The single most important thing to understand about the H100 is that it represents a **paradigm shift from synchronous to asynchronous execution**. Every previous GPU architecture required the SM's threads to babysit data movement — issuing loads, waiting for them, then computing. The H100 introduces hardware that does the data movement *in the background* while the SM computes on previously loaded data.

[![Serial work and pipelined overlap](images/ch01/05-sync-vs-async-model.svg)](images/ch01/05-sync-vs-async-model.svg)

*Figure 1.5 — Asynchronous copy is not unique to Hopper; TMA changes how bulk transfers are issued.*

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

[![A warp can work while data moves](images/ch01/06-warp-timeline.svg)](images/ch01/06-warp-timeline.svg)

*Figure 2.1 — Illustrative scheduling, not a measured utilisation trace.*

### 2.2 The TMA Unit

Each SM in the H100 has a dedicated **Tensor Memory Accelerator (TMA)** unit. This is a hardware block — it has its own logic, its own connection to the memory hierarchy, and its own understanding of multi-dimensional tensor layouts.

[![From per-thread copies to a tensor request](images/ch01/07-tma-before-after.svg)](images/ch01/07-tma-before-after.svg)

*Figure 2.2 — The transfer request defines the division of work between threads and hardware.*

### 2.3 TMA Descriptor — Concrete Example

The previous section described the TMA in the abstract. Here is the same copy
written both ways as conceptual pseudocode. Compare which address-generation and transfer tasks the threads perform; exact instruction counts require inspecting the compiled implementation.

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

  Each lane visits 128 elements in this scalar sketch.
  Thread-level operations are not summed to obtain warp-issued instructions.
  Actual instruction counts depend on vectorisation and compiler lowering.


TMA approach (single thread):
──────────────────────────────────────────────────────────
  // Thread 0 issues one instruction:
  tma_copy_2d(descriptor, /*box_coord=*/{256, 128}, smem_ptr, barrier);
  // Other threads may do independent work; consumers still wait for readiness.

  // The descriptor was set up once at kernel launch:
  //   base_ptr    = &matrix[0][0]
  //   dims        = {4096, 4096}
  //   global_dims = {4096, 4096}   (fastest dimension first)
  //   global_strides = {4096 * 2} (outer stride; inner BF16 size is implicit)
  //   box_size    = {64, 64}
  //   element_sz  = 2 bytes

  One tensor-copy request describes 4,096 BF16 elements = 8 KiB.
  Descriptor setup, issuing-thread election, barriers and waits add instructions.
  Hardware transaction counts depend on the memory layout and request handling.
```

### 2.4 The Bookkeeping Cost

The tradeoff for async execution is complexity. You need barriers to know when data is ready:

[![Two buffers, one repeating protocol](images/ch01/08-async-pipeline-phases.svg)](images/ch01/08-async-pipeline-phases.svg)

*Figure 2.3 — Never load a new tile over data that a consumer still needs.*

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
  4 tensor cores/SM working together via WGMMA.

  Dense throughput per SM, at the applicable Tensor Core clock:
    FP16/BF16: 4,096 FLOP/cycle/SM
    FP8:       8,192 FLOP/cycle/SM
    (An FMA counts as two FLOPs.)

  H100 SXM Tensor Core peak uses 1.83 GHz, not the 1.98 GHz
  CUDA-core boost clock. Mixing the clock specifications produces
  incorrect noninteger per-cycle rates.
    FP16/BF16: 4,096 × 132 × 1.83 GHz = 989.43 TFLOP/s dense
    FP8:       8,192 × 132 × 1.83 GHz = 1,978.86 TFLOP/s dense
    FP8 sparse effective peak: 3,957.72 TFLOP/s.

  H100 PCIe has 114 SMs and a 1.62 GHz Tensor Core boost:
    FP16/BF16: 4,096 × 114 × 1.62 GHz = 756.45 TFLOP/s dense.
  Actual sustained clocks depend on power, temperature and workload.

```

Sources: [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) and [NVIDIA’s Tensor Core throughput calculation](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727/2).

### 3.2 Warp Group MMA (WGMMA)

The H100 introduces **Warp Group** operations, where 4 warps (128 threads) cooperate on a single matrix multiply:

[![Where WGMMA operands live](images/ch01/09-wgmma-register-layout.svg)](images/ch01/09-wgmma-register-layout.svg)

*Figure 3.1 — Four warps cooperate on one warpgroup-level matrix operation.*

### 3.3 Structured Sparsity

Sparsity is a way of buying throughput by agreeing not to store some of your
numbers. The hardware supports exactly one pattern — "2:4" — and the payoff is
simple: the tensor core reads half as much data, so it runs about twice as fast.

[![Two values survive each quartet](images/ch01/10-structured-sparsity.svg)](images/ch01/10-structured-sparsity.svg)

*Figure 3.2 — 2:4 structured sparsity stores values plus position metadata.*

### 3.4 Precision Comparison

Lower precision buys throughput: the fewer bits per number, the more numbers a
tensor core can chew through per cycle. This table is the menu. The pattern to
notice is that halving the bit width roughly doubles the FLOPs — which is the
entire reason FP8 exists.

```
Precision     Bits  Range          Tensor Core FLOPS/cycle/SM*   Use Case
──────────────────────────────────────────────────────────────────────────
FP64          64    ±1.8×10^308    ~256                          Scientific
TF32          32**  ±3.4×10^38     2,048                        Training
FP32          32    ±3.4×10^38     (via CUDA cores only, 256)    General
FP16          16    ±65,504        4,096                        Training
BF16          16    ±3.4×10^38     4,096                        Training
FP8 (E4M3)     8    ±448           8,192                        Inference
FP8 (E5M2)     8    ±57,344        8,192                        Gradients
──────────────────────────────────────────────────────────────────────────
* Dense (no 2:4 sparsity), per SM per applicable execution clock.
  The FP16/BF16 and FP8 rows use the Tensor Core clock; do not divide
  their published TFLOP/s peaks by the CUDA-core boost clock.
  Sparse rates are effective dense-equivalent work for supported
  sparse instruction/type combinations; there is no sparse FP64 row.
** TF32 uses 32 bits of storage but a 10-bit mantissa internally.

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

[![Inside the Hopper GPU](images/ch01/01-die-floorplan.svg)](images/ch01/01-die-floorplan.svg)

*Figure 1.1 — Logical GH100 organisation. Blocks are schematic, not physical die placement.*

> **On the 132-of-144 figure:** the full GH100 die really does have 8 GPCs
> of 18 SMs each (144 total) [NVIDIA Hopper Architecture whitepaper, GTC22,
> v1.02]. The SXM5 product binned down to 132 SMs, described in the same
> whitepaper as "8 GPCs, 66 TPCs" -- 66 TPCs ÷ 8 GPCs is not a whole number,
> so the 12 disabled SMs are not spread evenly as "1.5 per GPC." NVIDIA does
> not publish the exact per-GPC enabled-SM breakdown for SXM5; treat the
> 18-SMs-per-GPC diagram above as the full-die floor plan, not a literal
> picture of which SMs are switched on in a given chip.

### 4.2 The GigaThread Engine

The GigaThread Engine is the hardware scheduler that sits at the top of the GPU and manages the entire kernel launch lifecycle:

[![How a grid becomes resident work](images/ch01/11-gigathread-flow.svg)](images/ch01/11-gigathread-flow.svg)

*Figure 4.2 — A logical launch path; exact scheduler implementation is not an ISA contract.*

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
about 33 TB/s GPU-wide modeled shared-memory bandwidth, an illustrative 12 TB/s aggregate L2 rate, and the 3.35 TB/s SXM HBM product peak. The first two are models or workload-dependent estimates, not product guarantees. Every step
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
```

[![The path from an SM to HBM](images/ch01/12-hbm-data-path.svg)](images/ch01/12-hbm-data-path.svg)

*Figure 5.1 — Cache hits can terminate a request before it reaches external memory.*

### 5.2 Why 5120-bit Bus Width Matters for TMA

A number like "5,120-bit bus" only becomes useful once you convert it into how
long your tile actually takes to arrive. That's what this does.

```
Bus width: 5120 bits = 640 bytes (10 × 512-bit memory controllers)
[NVIDIA Hopper Architecture whitepaper, GTC22, v1.02]

TMA can move 128 bytes of data at once in a single transaction.
This fits cleanly into the bus width:
  5120 bits / 8 = 640 bytes wide
  640 / 128 = 5 concurrent 128-byte transactions, bus-width-wise

For a 64×64 BF16 tile:
  64 × 64 × 2 bytes = 8,192 bytes

At the official aggregate HBM3 bandwidth of 3.35 TB/s
[NVIDIA H100 datasheet, PB-11773-001]:
  8,192 bytes / 3.35 TB/s ≈ 2.4 ns for one tile transfer once the
  request is in flight -- fast enough to keep tensor cores fed.

  Note: HBM3's own per-pin signaling clock runs far higher than, and
  in a completely separate clock domain from, the SM's 1,980 MHz boost
  clock. An earlier draft of this section estimated tile-transfer time
  from an assumed "~1.8 GHz memory clock," which conflated the two
  domains. This version derives the same number from the published
  aggregate bandwidth instead, so it doesn't need that assumption.
```

---

## 6 — L2 Cache

The L2 is a 50 MB staging area shared by every SM, and essentially all traffic
to and from HBM passes through it.

Two facts about it drive real kernel performance: it is organised into 128-byte
lines made of four 32-byte **sectors**, and it charges you per *sector touched*,
not per useful byte. That single detail is what "coalescing" means, and it is
why the same useful payload can generate very different transaction counts.
A transaction ratio is not itself a measured runtime ratio.

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
  Capacity model:    If 50 MiB is assumed, 50 × 2^20 / 128 = 409,600 lines

  Key behaviors:

  1. SECTOR-BASED FILLING
     A cache line is 128 bytes, but consists of 4 × 32-byte sectors.
     A memory request can touch 1-4 sectors of a single line.
     The cache can fill sector-by-sector (no need to load all 128 bytes).

  2. UNCOALESCED ACCESS PENALTY
     A warp of 32 threads issues a memory instruction.
     If threads access scattered addresses:
       Best case:  1 sector  (all 32 threads in same 32B)
       Worst case: 32 sectors (every thread in a different sector)

     Uncoalesced access explodes the number of sectors requested,
     consuming L2 bandwidth and evicting useful cache lines.

  3. CACHE LOCALITY
     L2 is shared across SMs. Address mapping and concurrent traffic affect
     locality and contention; the program cannot assume an SM's working
     set is automatically assigned to a nearby partition.

  4. WRITE COALESCING
     Store transactions and L2 caching affect downstream memory traffic.
     Coalescing is based on lane addresses; cache behavior does not turn
     every scatter/atomic request into a contiguous HBM transfer.
```

[![A 128-byte cache line has four sectors](images/ch01/13-l2-sectors.svg)](images/ch01/13-l2-sectors.svg)

*Figure 6.1 — Sector granularity helps explain coalescing and overfetch.*

### 6.2 Worked Example — L2 Sector Math

Same useful payload, two access patterns, an eightfold difference in requested sectors. Work through
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
Requested footprint: four sectors within one cache line

Result: COALESCED — four 32-byte sectors cover the 128 useful bytes.


Bad scenario: 32 threads each load a float with stride 128 bytes

Thread  0: 0x1000
Thread  1: 0x1080  (128 bytes later)
Thread  2: 0x1100
...
Thread 31: 0x1F80

Total bytes still 128, but spread across 32 different cache lines.
Sectors touched: 32 (one sector per thread, all different lines)
Requested footprint: 32 sectors spread across 32 cache lines.

Result: eight times as many sectors (32 versus 4), 1,024 requested bytes
for the same 128 useful bytes. Cache hits affect downstream HBM traffic;
this sector ratio is not a runtime ratio.
```

---

## 7 — GPCs — Graphics Processing Clusters

The enabled SMs are grouped into **Graphics Processing Clusters**. The full
GH100 die has eight GPCs with 18 SMs each; the 132-SM SXM product disables
some resources, so its enabled GPCs do not all contain 18 active SMs.

For most of GPU history this grouping was invisible to programmers. On Hopper it
stopped being invisible, because *thread block clusters* — groups of blocks that
can read each other's shared memory — only work within a single GPC. This is a
short section, but it is the reason cluster sizes cap out where they do.

### 7.1 GPC Internal Structure

One GPC, drawn out. The pieces to notice: the 18 SMs, their pairing into TPCs,
and the DSMEM interconnect that lets SMs read one another's shared memory
directly — without going out to L2 and back.

In the full GH100 configuration, one GPC contains 18 SMs organized as nine TPCs of two SMs each. Product configurations enable fewer resources. Clusters provide distributed shared-memory access within a GPC; L2 is a GPU-wide resource.

[![A GPC contains multiple TPCs](images/ch01/14-gpc-internal.svg)](images/ch01/14-gpc-internal.svg)

*Figure 7.1 — Full-die schematic: nine texture processing clusters, two SMs per TPC.*

### 7.2 Why GPCs Matter for Kernel Design

Source: [NVIDIA Hopper tuning guide cluster and occupancy limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

The practical consequence of everything above: how large can a cluster be, and
what does each size cost you in synchronisation?

```
Implication for Thread Block Clusters:
═══════════════════════════════════════

  A cluster of thread blocks must all run on SMs within the SAME GPC.
  (Because clusters need DSMEM, and DSMEM only exists within a GPC.)

  Portable maximum: eight blocks, subject to device/configuration limits.
  Larger clusters require nonportable opt-in and actual device support.
  Query cudaOccupancyMaxPotentialClusterSize and cudaOccupancyMaxActiveClusters.

  The 18-SM full-die GPC diagram does not establish a cluster-size limit,
  blocks-per-SM placement rule, active-cluster count or synchronization cost.
  Registers, shared memory, warps, enabled SMs and cluster shape all matter.

```

---

## 8 — NVLink 4.0

Everything so far has been inside a single GPU. NVLink is the wire *between*
GPUs — 900 GB/s aggregate bidirectional on SXM, roughly seven times the bidirectional PCIe 5.0 ×16 link rate. A one-way comparison is 450 versus approximately 64 GB/s before protocol overhead.

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
  Links:               18 fourth-generation NVLink links
  Per link:            50 GB/s bidirectional
                        (25 GB/s per direction, via 2 differential
                        signal pairs per direction)
  Check:               18 × 50 GB/s = 900 GB/s  ✓
  [NVIDIA Hopper Architecture whitepaper, GTC22, v1.02]

  Signaling:           PAM4 (Pulse Amplitude Modulation, 4 levels)
  ┌─────────────────────────────────────────────────────────┐
  │  NRZ  (NVLink 1.0-3.0): 2 amplitude levels → 1 bit/symbol │
  │  PAM4 (NVLink 4.0+):    4 amplitude levels → 2 bits/symbol│
  │                                                            │
  │  Same baud rate, twice the bits per symbol →              │
  │  2× the data rate per differential pair.                  │
  │  (NVIDIA does not publish exact per-level voltages --      │
  │  only the 4-level/2-bit-per-symbol scheme is public.)      │
  └─────────────────────────────────────────────────────────┘

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

[![Eight GPUs, one NVSwitch fabric](images/ch01/04-dgx-topology.svg)](images/ch01/04-dgx-topology.svg)

*Figure 1.3 — DGX H100 connects all eight GPUs through a switched NVLink network. [NVIDIA HGX H100 platform comparison](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/)*

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

[![One SM, four processing partitions](images/ch01/02-sm-internals.svg)](images/ch01/02-sm-internals.svg)

*Figure 1.2 — Each SMSP schedules warps and uses a share of the SM’s execution resources.*

---

## 10 — SM Quadrants (SMSPs)

An SM is not one processor — it is four, and they schedule independently.

Block size affects the available warps, register use and resident blocks. Multiple resident blocks can supply work to the SM’s schedulers, so a block size that is not a multiple of 128 does not by itself imply an idle SMSP. §10.2 uses an illustrative distribution; profile the actual kernel before drawing a utilisation conclusion.

### 10.1 Quadrant Breakdown

Each SM is divided into 4 identical subdivisions. Each quadrant is an independently scheduling unit:

[![A processing partition, closer up](images/ch01/15-smsp-quadrant-detail.svg)](images/ch01/15-smsp-quadrant-detail.svg)

*Figure 10.1 — SMSPs make warp scheduling and register pressure easier to reason about.*

### 10.2 Why 4 Quadrants?

Here is the practical payoff of knowing the SM is four things rather than one.

[![A block is a collection of warps](images/ch01/16-thread-block-distribution.svg)](images/ch01/16-thread-block-distribution.svg)

*Figure 10.2 — 256 threads make eight warps of 32 threads each.*

The [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) specifies the per-SM register and residency limits. Its 64K 32-bit registers are 256 KiB per SM, corresponding to 16,384 registers in each 64 KiB partition shown here.

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

[![What makes a warp eligible to issue?](images/ch01/17-warp-scheduler-cycle.svg)](images/ch01/17-warp-scheduler-cycle.svg)

*Figure 11.1 — A conceptual dependency-to-execution loop, not a hardware pipeline timing diagram.*

### 11.3 The Dispatch Unit and Dispatch Ports

This is the one part of the SM that no occupancy calculator will show you, and
it explains why "the SM has 128 FP32 cores and 64 INT32 units" does not mean a
single warp can use both in the same cycle.

[![Issue bandwidth and execution bandwidth](images/ch01/18-dispatch-ports.svg)](images/ch01/18-dispatch-ports.svg)

*Figure 11.2 — An instruction must be issued before an execution unit can do useful work.*

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
  When a warp executes a load, 32 threads each provide an address.
  The LSU attempts to MERGE these 32 addresses into as few
  cache-line requests as possible.
```

[![Coalescing: the same payload, different traffic](images/ch01/19-coalescing.svg)](images/ch01/19-coalescing.svg)

*Figure 12.1 — A warp’s lane addresses determine which sectors it touches.*

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
  Bandwidth:      ~0.25 TB/s per SM (32 banks × 4 B/cycle × 1.98 GHz model)
  Banks:          32 banks × 4 bytes (32 bits) wide
  Cache line:     128 bytes
  Sector:         32 bytes (cache fills sector-by-sector)

  The 256 KB is shared between:
    ├── Shared memory (programmer-managed, explicit)
    └── L1 data cache (hardware-managed, automatic)
```

[![Shared memory and L1 share a resource](images/ch01/20-shared-mem-split.svg)](images/ch01/20-shared-mem-split.svg)

*Figure 13.1 — H100 has a unified 256 KiB resource; supported carveouts are constrained by the product and kernel.*

```

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

[![Shared-memory bank mapping](images/ch01/21-shared-mem-banks.svg)](images/ch01/21-shared-mem-banks.svg)

*Figure 13.2 — For ordinary four-byte words, addresses interleave across 32 banks.*

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

[![A swizzle changes the physical bank pattern](images/ch01/22-swizzling.svg)](images/ch01/22-swizzling.svg)

*Figure 13.3 — A small XOR permutation makes column-wise access easier to see.*

```
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

  CUDA 13.0 opt-in feature (not used by the CUDA 12.8 examples here):
    In supported whole-program compilation, enable_smem_spilling permits
    PTXAS to use shared memory first, with remaining spills in local memory.
    It is disabled by default and consumes shared-memory capacity; measure
    both spill traffic and occupancy before assuming a performance gain.

  How to detect spilling:
    nvcc --ptxas-options=-v  →  shows "spill loads" and "spill stores"
    If you see any, your kernel is likely register-pressure limited.

  How to reduce register usage:
    ├── __launch_bounds__(maxThreads, minBlocks)
    ├── Recompute values instead of storing them
    ├── Pack compatible FP16 pairs when supported; scalar FP16 alone does
    │   not guarantee half the physical register allocation
    └── Restructure code to reduce live variables
```

Source: [NVIDIA shared-memory register spilling requirements](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/).

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

Follow one thread block from placement through warp scheduling. Each block remains assigned to an SM; warps can execute independently, so `__syncthreads()` coordinates all participating threads in that block regardless of the illustrated partition placement.

[![From thread block to SM execution](images/ch01/23-cta-to-sm-flow.svg)](images/ch01/23-cta-to-sm-flow.svg)

*Figure 16.1 — Resource requirements constrain residency; synchronisation controls correctness.*

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

[![A GEMM tile through the memory system](images/ch01/24-gemm-tile-dataflow.svg)](images/ch01/24-gemm-tile-dataflow.svg)

*Figure 18.1 — The main loop and epilogue have different dataflow and lifetime requirements.*

### 18.2 Memory Hierarchy Summary

The hierarchy on one card. The right-hand columns are the ones worth
internalising: each level down costs roughly an order of magnitude more latency
and 10× more energy per access.

```
H100 Memory Hierarchy — Bandwidth vs Capacity
══════════════════════════════════════════════════════════

  Level          Size        Bandwidth/model       Latency (illustrative)
  ──────────────────────────────────────────────────────────────
  Registers     256 KB/SM    ~128 R/W/cycle   operand access; instruction-dependent
  Shared Mem    228 KB/SM    ~0.25 TB/s/SM      ~20 cycles, varies
  L1 Cache      28-224 KB    (part of shared) ~30 cycles, varies
  L2 Cache      50 MB        ~12 TB/s agg    ~200 cycles, varies
  HBM3          80 GB        3.35 TB/s       ~500 cycles, varies
  ──────────────────────────────────────────────────────────────

  The ratio:
    HBM3 → L2:    ~3.5×  bandwidth amplification
    L2   → SMEM:  ~2.8× GPU-wide (modeled 33 TB/s SMEM vs illustrative 12 TB/s L2)
    SMEM → Regs: no universal bandwidth ratio; operand width and reuse matter

  Each level filters traffic for the next:
    If L1 hits 90%, only 10% of requests reach L2.
    If L2 hits 80%, only 20% of L2 requests reach HBM.
    Total HBM traffic = 10% × 20% = 2% of original requests.

  This cascading filtering is what makes the hierarchy work.
```

[![Memory hierarchy: scope matters](images/ch01/03-memory-hierarchy.svg)](images/ch01/03-memory-hierarchy.svg)

*Figure 1.4 — Capacity, product peak bandwidth and a bank-throughput model are different quantities.*

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
Shared Mem (max)    228 KB/SM    ~0.25 TB/s/SM     32 banks × 4B
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

Tensor Core throughput (SXM5, derived from published full-GPU totals):
  FP16/BF16: 4,096 FLOPS/cycle/SM (dense)
  FP8:       8,192 FLOPS/cycle/SM (dense)
  Sparse (2:4): 2× dense throughput
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

<!-- practice:start -->
**Pause and explain it.** These are worked reasoning exercises, not additional hardware execution claims. Attempt each question before opening its answer.

**Self-check 1.1.** A GPU link is advertised as 900 GB/s bidirectional. What one-way budget may you use in a simple symmetric model?

??? success "Worked answer — 1.1"

    450 GB/s in each direction. Do not compare the 900 GB/s bidirectional sum with a one-way network rate. An eight-port aggregate is also a different quantity from a four-versus-four bisection.

**Self-check 1.2.** Assume 32 shared-memory banks, four bytes per bank per cycle and a 1.98 GHz clock. Compute the modeled per-SM rate.

??? success "Worked answer — 1.2"

    32 × 4 × 1.98×10⁹ = 253.44×10⁹ bytes/s, or 253.44 GB/s per SM. Multiplying by 132 SMs gives about 33.45 TB/s GPU-wide. This is an assumed bank-throughput model, not a measured bandwidth guarantee.

<!-- practice:end -->


---

> **Next — Chapter 2: Clusters, Inline PTX, Data Types & Pointers**
>
> You now know what the hardware is. Chapter 2 covers the programming model that
> reaches it: thread block clusters and distributed shared memory, inline PTX
> assembly, state spaces, the deep-learning number formats, and the
> address-conversion instructions (`cvta`, `mapa`) that later chapters lean on
> constantly.
