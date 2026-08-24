# FlashAttention-3 on the H100
## Chapter 12 Companion Guide — Every Primitive in This Book, Working at Once

> **Reference implementation — not executed during authoring.** Unlike Chapters
> 1–11, the code in this chapter has not been compiled and run on Hopper
> hardware. It is written to be correct and to be read, and the structure follows
> the published FlashAttention-3 design, but treat it as a annotated blueprint
> rather than a tested artifact.
> **Target:** `nvcc -std=c++17 -arch=sm_90a -O2 <file>.cu -o <out>`
> **Canonical source:** Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao —
> *FlashAttention-3: Fast and Accurate Attention with Asynchrony and
> Low-precision*, NeurIPS 2024 ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608)).

**What this covers:** Why attention is memory-bound in its naive form and how
tiling fixes it; the online softmax recurrence that makes tiling possible, worked
by hand; what each generation of FlashAttention fixed; and then the three Hopper
techniques that take FA3 from FlashAttention-2's 35% utilization to 75% —
producer/consumer warp specialization, ping-pong scheduling across two
warpgroups, and 2-stage pipelining inside a single warpgroup. Then causal masking
and its load-imbalance problem, and the FP8 path: block quantization, the K-major
layout constraint that forces an in-kernel transpose of V, and the Hadamard
"incoherent processing" trick that recovers accuracy.

**How to read this chapter.** Everything up to now has been about primitives in
isolation — a descriptor here, a barrier there, one `wgmma` issued correctly.
This chapter is the payoff: a single kernel where all of them run at once, on a
problem people actually care about.

It splits in three. Sections 1–3 are algorithmic and hardware-independent —
what attention is, why the naive form wastes memory bandwidth, and the online
softmax recurrence that lets you tile it. Sections 4–8 are the Hopper kernel:
three techniques, applied one at a time, with the measured throughput after each.
Sections 9–10 cover FP8, which is a separate discipline layered on top.

If you have read Chapters 8 and 9, Sections 5–7 will feel familiar — they are
warp specialization and the ping-pong pipeline you already know, applied to a
problem with an awkward extra step in the middle. That awkward step, softmax, is
the whole story of this chapter.

---

## ★ BEFORE YOU START — What You Need First

### Prerequisites

This chapter assumes more of the book than any other. It does not re-explain the
primitives it uses.

```
Prerequisites Checklist
══════════════════════════════════════════════════════════════════════

  From Chapter 3 (mbarrier):
    ✓ mbarrier init / arrive / expect_tx / try_wait
    ✓ Phase bits and barrier reuse in a loop
    ✓ The two-barrier handshake (data ready + buffer free)

  From Chapters 4-5 (TMA):
    ✓ cuTensorMap descriptors and the dimension-ordering rule
    ✓ cp.async.bulk.tensor with mbarrier completion
    ✓ Swizzling (128B mode) and why it exists

  From Chapters 6-7 (WGMMA):
    ✓ wgmma.mma_async, warp groups, commit/wait groups
    ✓ The accumulator register layout across 128 threads
    ✓ ldmatrix / stmatrix
    ✓ FP8 formats, packed types, cvt.satfinite

  From Chapters 8-9 (kernel design):
    ✓ Producer/consumer warp specialization
    ✓ setmaxnreg
    ✓ Circular buffering with N stages
    ✓ The ping-pong pipeline architecture
    ✓ Named barriers

  New here (no prior knowledge needed):
    ★ The attention operation and its arithmetic intensity
    ★ Online softmax — the running-max recurrence
    ★ Interleaving a non-tensor-core operation into a tensor-core pipeline
    ★ Causal masking and triangular load imbalance
    ★ Block quantization and incoherent processing
```

### Hardware

Same as Chapters 6–11: Hopper (`sm_90a`) or later. Nothing in this chapter has a
pre-Hopper equivalent — the entire design exists because of TMA, `wgmma`, and
`setmaxnreg`.

### Reading the Real Implementation

The production code lives in the
[`flash-attention`](https://github.com/Dao-AILab/flash-attention) repository
under `hopper/`. It is written in CUTLASS/CuTe rather than raw PTX, which makes
it dense on first contact but much shorter than the equivalent hand-written
kernel.

```
Mapping This Chapter to the Real Source
══════════════════════════════════════════════════════════════════════

  This chapter's §      flash-attention/hopper/
  ────────────────────────────────────────────────────────────────────
  §5  warp specialization   flash_fwd_kernel.h — the WarpScheduler split
  §6  ping-pong             mainloop_fwd_sm90_tma_gmma_ws.hpp
  §7  2-stage pipeline      the same file — see the "PV" / "QK" ordering
  §9  FP8                   flash_fwd_kernel.h with Element = cutlass::float_e4m3_t
  §9.2 V transpose          the SmemTransposeFp8 helper

  Read this chapter first for the ideas, then that code for the details.
  The CuTe layout algebra is a language of its own; do not expect to
  read it cold.
```

---

## Table of Contents

1. [Why Attention Is Hard on a GPU](#1--why-attention-is-hard-on-a-gpu)
2. [Online Softmax](#2--online-softmax)
3. [The FlashAttention Family](#3--the-flashattention-family)
4. [The FA3 Kernel at a Glance](#4--the-fa3-kernel-at-a-glance)
5. [Technique 1 — Producer/Consumer Warp Specialization](#5--technique-1--producerconsumer-warp-specialization)
6. [Technique 2 — Ping-Pong Scheduling](#6--technique-2--ping-pong-scheduling)
7. [Technique 3 — Intra-Warpgroup 2-Stage Pipelining](#7--technique-3--intra-warpgroup-2-stage-pipelining)
8. [Causal Masking and Load Imbalance](#8--causal-masking-and-load-imbalance)
9. [FP8 FlashAttention](#9--fp8-flashattention)
10. [Putting It All Together](#10--putting-it-all-together)
11. [Measured Results](#11--measured-results)
12. [Quick Reference](#12--quick-reference)

---

## 1 — Why Attention Is Hard on a GPU

Attention looks, on paper, like two matrix multiplies with a softmax in the
middle. Two matrix multiplies is exactly what this hardware is built for. So why
did it take three papers and four years to make it fast?

The answer is the thing in the middle. Section 1.1 states the operation, §1.2
shows what the naive implementation costs in memory traffic, and §1.3 puts a
number on it with the roofline model from Chapter 8.

### 1.1 The Attention Operation

Given three matrices — queries `Q`, keys `K`, values `V` — attention computes:

```
Attention — The Operation
══════════════════════════════════════════════════════════════════════

  S = Q K^T / sqrt(d)        scores       [N × N]
  P = softmax(S)             probabilities [N × N]   (row-wise)
  O = P V                    output        [N × d]

  Shapes for one head:
    Q:  [N × d]      N = sequence length (512 … 128k)
    K:  [N × d]      d = head dimension  (64, 128, or 256)
    V:  [N × d]
    O:  [N × d]

  A real model runs this for every head of every layer:
    Llama-3-70B: 64 heads × 80 layers = 5,120 attention calls
    per forward pass, per sequence.

  FLOP count (forward, one head):
    QK^T :  2 N² d
    PV   :  2 N² d
    ─────────────────
    total:  4 N² d        (softmax is O(N²) but not FLOP-dominant)
```

The quadratic term is the problem and also the point: every token attends to
every other token, so the score matrix `S` is `N × N`.

### 1.2 The Materialization Problem

Write that down naively — three separate kernels, or three separate library
calls — and you get the following.

```
Naive Attention — Where the Bytes Go
══════════════════════════════════════════════════════════════════════

  N = 8192, d = 128, BF16 (2 bytes/element), one head

  Step 1: S = Q K^T
    read  Q:  8192 × 128 × 2  =    2.1 MB
    read  K:  8192 × 128 × 2  =    2.1 MB
    WRITE S:  8192 × 8192 × 2 =  134.2 MB   ◄── to HBM
                                 ─────────
                                  138.4 MB

  Step 2: P = softmax(S)
    read  S:  134.2 MB
    WRITE P:  134.2 MB                       ◄── to HBM
              ─────────
              268.4 MB

  Step 3: O = P V
    read  P:  134.2 MB
    read  V:    2.1 MB
    WRITE O:    2.1 MB
              ─────────
              138.4 MB

  ══════════════════════════════════════════════════════════════════
  TOTAL HBM TRAFFIC:  545.2 MB
  USEFUL INPUT/OUTPUT: 8.4 MB  (Q, K, V in; O out)

  Ratio: 65× more traffic than the problem actually requires.
  ══════════════════════════════════════════════════════════════════

  And the capacity problem is worse than the bandwidth problem:
    S alone is 134 MB. The H100's entire L2 is 50 MB.
    S cannot be cached. Every byte written comes back from HBM.

  Scaling: S grows as N². At N = 32k it is 2.1 GB per head.
```

The whole of FlashAttention rests on one observation: **`S` and `P` never need to
exist in HBM.** They are intermediates. If you can compute the output tile by
tile, keeping each `S` tile in shared memory and throwing it away, the traffic
collapses to the 8.4 MB you actually needed.

### 1.3 Arithmetic Intensity — Why Tiling Wins

Chapter 8 §1 defined arithmetic intensity as FLOPs per byte of HBM traffic, and
the roofline as the line where that number stops mattering. Apply it here.

```
Arithmetic Intensity — Naive vs Tiled
══════════════════════════════════════════════════════════════════════

  H100 ridge point (Chapter 8 §1):
    989 TFLOP/s (BF16 tensor) ÷ 3.35 TB/s = ~295 FLOP/byte
    Below 295 FLOP/byte you are memory-bound. Above it, compute-bound.

  NAIVE (N=8192, d=128):
    FLOPs  = 4 N² d       = 4 × 8192² × 128  = 34.4 GFLOP
    Bytes  = 545.2 MB
    Intensity = 34.4e9 / 545.2e6 = 63 FLOP/byte

    63 << 295  →  MEMORY BOUND, by a factor of ~4.7
    Ceiling: 3.35 TB/s × 63 = 211 TFLOP/s ... but you never
    get close, because the access pattern is also poor.

  FLASH (fused, S never leaves SMEM):
    FLOPs  = 34.4 GFLOP  (unchanged — same math)
    Bytes  = 8.4 MB      (Q, K, V read once; O written once)
    Intensity = 34.4e9 / 8.4e6 = 4095 FLOP/byte

    4095 >> 295  →  COMPUTE BOUND, by a factor of ~14

  ══════════════════════════════════════════════════════════════════
  Fusing attention moves it across the ridge point.
  That is the entire contribution of FlashAttention-1.
  Everything after it is about how well you exploit being
  compute-bound once you are there.
  ══════════════════════════════════════════════════════════════════
```

That last line is worth sitting with. FlashAttention-1 made attention
compute-bound. FlashAttention-2 and 3 are not about memory at all — they are
about the fact that being compute-bound is necessary but nowhere near sufficient.
FA2 on an H100 is compute-bound *and* runs at 35% of peak.

---

## 2 — Online Softmax

There is an obstacle to the plan in §1.2, and it is not obvious until you try.

Softmax needs a row maximum and a row sum. Both are reductions over the **entire
row** of the score matrix. But tiling means processing one column-block of that
row at a time — so when you handle the first block, you do not yet know the row
maximum, and you cannot compute the softmax.

Online softmax is the recurrence that dissolves this. It is a small piece of
numerical trickery, it is exact rather than approximate, and it is the reason the
entire FlashAttention family exists.

### 2.1 Softmax and Why It Resists Tiling

```
Softmax — The Standard Numerically-Stable Form
══════════════════════════════════════════════════════════════════════

  For a row vector x of length N:

    m     = max(x)                    ◄── pass 1 over the whole row
    p_i   = exp(x_i - m)              ◄── pass 2
    l     = sum(p_i)                  ◄── pass 2
    out_i = p_i / l                   ◄── pass 3

  Why subtract m?
    exp() overflows FP32 above ~88. Attention scores routinely
    exceed that. Subtracting the max makes the largest exponent
    exactly exp(0) = 1, so nothing overflows and the result is
    mathematically identical:
        exp(x_i - m) / Σ exp(x_j - m)  ==  exp(x_i) / Σ exp(x_j)

  The problem for tiling:
    m depends on ALL of x. So does l.
    A tiled loop sees x in chunks and cannot compute either
    until it has seen the last chunk.

    Naive fix: two passes over K/V. That doubles the memory
    traffic you just worked so hard to eliminate.
```

### 2.2 The Running Maximum and the Rescaling Trick

The insight: you can compute softmax with a *running* maximum, provided you go
back and correct what you have already accumulated whenever the maximum changes.

And correcting is cheap — because `exp` turns a change in the max into a
multiplicative factor.

```
Online Softmax — The Recurrence
══════════════════════════════════════════════════════════════════════

  State carried between blocks, per row:
    m   running maximum       (init: -inf)
    l   running sum of exp    (init: 0)
    O   running output accum  (init: 0)     [d-dimensional]

  For each new score block S_j:

    1. m_new = max(m_old, rowmax(S_j))

    2. α = exp(m_old - m_new)         ◄── the correction factor
                                          α ≤ 1 always
    3. P_j = exp(S_j - m_new)         ◄── this block's probabilities

    4. l_new = α · l_old + rowsum(P_j)

    5. O_new = α · O_old + P_j V_j    ◄── rescale, then accumulate

  At the very end, once:
    O = O / l

  ══════════════════════════════════════════════════════════════════
  WHY IT WORKS

  Everything accumulated so far was computed relative to m_old.
  The new maximum is m_new. Every old term is off by exactly

      exp(x - m_old) / exp(x - m_new) = exp(m_new - m_old)

  so multiplying by α = exp(m_old - m_new) converts every
  previously-accumulated term to the new reference — in one
  multiply, without revisiting any of them.

  This is EXACT. Not an approximation, not a bound. The result is
  bit-comparable to two-pass softmax up to floating-point ordering.
  ══════════════════════════════════════════════════════════════════

  The cost:
    ├── One extra multiply per element of O, per block  (step 5)
    ├── Two extra multiplies per row, per block          (steps 2, 4)
    └── That is all. No extra memory traffic whatsoever.
```

### 2.3 Worked Example — Online Softmax by Hand

Do this once by hand and it will never be mysterious again.

```
Worked Example — One Row, Two Blocks
══════════════════════════════════════════════════════════════════════

  Row of scores (N = 4), processed as two blocks of 2:
    Block 0:  S = [1.0, 3.0]     V rows: v0 = 10, v1 = 20
    Block 1:  S = [5.0, 2.0]     V rows: v2 = 30, v3 = 40

  (V is 1-dimensional here so the arithmetic stays readable.)

  ─── Reference answer, two-pass ──────────────────────────────────
    m = max(1,3,5,2) = 5
    p = [exp(-4), exp(-2), exp(0), exp(-3)]
      = [0.01832, 0.13534, 1.00000, 0.04979]
    l = 1.20345
    softmax = [0.015223, 0.112459, 0.830943, 0.041373]
    O = 0.015223(10) + 0.112459(20) + 0.830943(30) + 0.041373(40)
      = 0.15223 + 2.24918 + 24.92829 + 1.65492
      = 29.0 (to 3 s.f.  → 28.98462)

  ─── Online, block by block ──────────────────────────────────────

  INIT:  m = -inf,  l = 0,  O = 0

  ── BLOCK 0:  S = [1.0, 3.0] ────────────────────────────────
    rowmax(S_0)  = 3.0
    m_new        = max(-inf, 3.0) = 3.0
    α            = exp(-inf - 3.0) = 0            ◄── first block
    P_0          = [exp(1-3), exp(3-3)]
                 = [exp(-2), exp(0)] = [0.13534, 1.00000]
    l            = 0·0 + (0.13534 + 1.00000) = 1.13534
    O            = 0·0 + (0.13534×10 + 1.00000×20)
                 = 1.35340 + 20.00000 = 21.35340

    State: m = 3.0,  l = 1.13534,  O = 21.35340

  ── BLOCK 1:  S = [5.0, 2.0] ────────────────────────────────
    rowmax(S_1)  = 5.0
    m_new        = max(3.0, 5.0) = 5.0            ◄── max INCREASED
    α            = exp(3.0 - 5.0) = exp(-2) = 0.13534
    P_1          = [exp(5-5), exp(2-5)]
                 = [exp(0), exp(-3)] = [1.00000, 0.04979]
    l            = 0.13534 × 1.13534 + (1.00000 + 0.04979)
                 = 0.15366 + 1.04979
                 = 1.20345                        ◄── matches reference l ✓
    O            = 0.13534 × 21.35340 + (1.00000×30 + 0.04979×40)
                 = 2.89000 + (30.00000 + 1.99160)
                 = 2.89000 + 31.99160
                 = 34.88160

  ── FINALIZE ────────────────────────────────────────────────
    O = 34.88160 / 1.20345 = 28.98462             ◄── matches ✓

  ══════════════════════════════════════════════════════════════════
  Note what happened in block 1. The old accumulator, 21.35340,
  had been built assuming a max of 3.0. One multiply by α = 0.13534
  retroactively rebased all of it to a max of 5.0 — without
  touching block 0's data, which by then was long gone from
  shared memory.

  That single multiply is why attention can be tiled.
  ══════════════════════════════════════════════════════════════════
```

**The one thing to notice for later:** step 5 rescales `O`, which is the
accumulator living in tensor core registers. That means every time the row max
moves, the kernel must reach into the WGMMA accumulator and multiply it. Chapter
7 §5 warned that touching accumulator registers has fence implications — this is
where that bites.

---

## 3 — The FlashAttention Family

Three papers, three distinct problems solved. Knowing which is which tells you
what FA3 is actually contributing.

### 3.1 What Each Generation Fixed

```
FlashAttention 1 → 2 → 3
══════════════════════════════════════════════════════════════════════

  FlashAttention-1 (2022)          Target: A100
  ────────────────────────────────────────────────────────────────────
  Problem:  attention materialises an N×N matrix in HBM
  Fix:      tile it; keep S in SMEM; online softmax (§2)
  Result:   memory traffic 545 MB → 8 MB; attention becomes
            compute-bound
  Left on the table: poor work partitioning, lots of non-matmul work

  FlashAttention-2 (2023)          Target: A100
  ────────────────────────────────────────────────────────────────────
  Problem:  too many non-matmul FLOPs; bad parallelism over
            sequence length; work split across warps causing
            SMEM round-trips
  Fix:      rescale O once at the end rather than every block where
            possible; parallelise over sequence-length blocks as
            well as heads/batch; split K/V across warps instead of
            Q ("split-K" → "split-Q")
  Result:   ~2× over FA1; ~70% of A100 peak
  Left on the table: EVERYTHING Hopper-specific

  FlashAttention-3 (2024)          Target: H100
  ────────────────────────────────────────────────────────────────────
  Problem:  FA2 on H100 hits ~35% of peak. The kernel is
            synchronous in a machine designed to be asynchronous.
  Fix:      (1) producer/consumer warp specialization with TMA
            (2) ping-pong scheduling across two warpgroups
            (3) 2-stage pipelining within a warpgroup
            (4) FP8 with block quantization + incoherent processing
  Result:   740 TFLOP/s BF16 (75% of peak), ~1.2 PFLOP/s FP8
```

### 3.2 Why FA2 Left 65% of the H100 Unused

This is the diagnosis that motivates the whole chapter, and it is worth stating
in the vocabulary of Chapters 3 through 9.

```
FA2 on H100 — The Diagnosis
══════════════════════════════════════════════════════════════════════

  FA2's inner loop, in H100 terms:

    for each K/V block:
        load K_j, V_j into SMEM      ◄── synchronous cp.async
        __syncthreads()              ◄── ALL warps wait
        S = Q K_j^T                  ◄── mma.sync (Ampere-style)
        softmax bookkeeping          ◄── SFU work, tensor cores IDLE
        O += P V_j                   ◄── mma.sync
        __syncthreads()

  Four things are wrong on Hopper, each costing double-digit percent:

  1. THE LOAD IS NOT FREE
     cp.async still burns SM instruction issue on address math.
     Chapter 5 §2: Hopper's cp.async.bulk offloads this entirely.

  2. EVERY WARP WAITS FOR EVERY WARP
     __syncthreads() is a whole-block rendezvous. Warps that
     could be computing are parked. Chapter 3 §16: named barriers
     let subsets synchronise independently.

  3. TENSOR CORES IDLE DURING SOFTMAX
     This is the big one. See below.

  4. mma.sync IS BLOCKING
     Chapter 6: wgmma.mma_async returns immediately. mma.sync
     does not. There is nothing to overlap with.

  ══════════════════════════════════════════════════════════════════
  THE SOFTMAX GAP — the number that explains FA3

  H100 SXM5 throughput:
    BF16 matmul (tensor cores):    989 TFLOP/s
    exp() (SFUs, Chapter 1 §12.4):   3.9 TFLOP/s
                                   ─────────────
    Ratio:                           ~256×

  Attention's FLOP mix is roughly:
    matmul:  4 N² d          for N=8192, d=128 → 34.4 GFLOP
    exp:     N²              →  67.1 MFLOP  (0.2% of the FLOPs)

  0.2% of the FLOPs. But at 256× lower throughput:

    matmul time:  34.4e9 / 989e12   =  34.8 µs
    exp    time:  67.1e6 / 3.9e12   =  17.2 µs
                                       ────────
    If SERIALISED:                     52.0 µs
    If OVERLAPPED:                     34.8 µs

    Serialising softmax costs 33% of your runtime for 0.2% of
    your work.
  ══════════════════════════════════════════════════════════════════

  That calculation is the entire motivation for §6 and §7.
  Both techniques exist to get the exp() off the critical path.
```

---

## 4 — The FA3 Kernel at a Glance

Before the three techniques arrive one at a time, here is the shape of the thing
they are being applied to.

### 4.1 The Three Techniques and What Each Buys

```
FA3 — Cumulative Throughput (BF16 forward, H100 SXM5)
══════════════════════════════════════════════════════════════════════

  Baseline: FlashAttention-2 on H100          ~350 TFLOP/s   (35%)
                                               │
  + Technique 1: warp specialization + TMA     │
    (producer warps fetch, consumer warps      ▼
     compute; §5)                             ~570 TFLOP/s   (58%)
                                               │
  + Technique 2: ping-pong across two          │
    warpgroups (§6)                            ▼
                                              ~620 TFLOP/s   (63%)
                                               │
  + Technique 3: 2-stage pipeline inside       │
    each warpgroup (§7)                        ▼
                                          640-660 TFLOP/s   (65-67%)
                                               │
  + tuning, tile shapes, epilogue              ▼
                                           up to 740 TFLOP/s (75%)

  FP8 path (§9):                          up to ~1.2 PFLOP/s

  ══════════════════════════════════════════════════════════════════
  Read the ladder carefully. Technique 1 — just using the
  hardware the way Chapters 5, 8 and 9 describe — is worth more
  than the two clever scheduling tricks combined.

  The exotic parts buy the last 15%. Getting the basics right
  buys the first 23 points.
  ══════════════════════════════════════════════════════════════════
```

### 4.2 Tile Shapes and the Two GEMMs

```
The Two GEMMs Inside Attention
══════════════════════════════════════════════════════════════════════

  Per Q-block, looping over K/V blocks:

  GEMM0 ("QK"):   S_j = Q_i · K_j^T
      Q_i : [B_r × d]     lives in registers or SMEM, reused
                          across the entire inner loop
      K_j : [B_c × d]     streamed in by TMA
      S_j : [B_r × B_c]   accumulator, FP32

  ── softmax on S_j (§2) → P_j, and rescale O_i ──

  GEMM1 ("PV"):   O_i += P_j · V_j
      P_j : [B_r × B_c]   just computed, in registers
      V_j : [B_c × d]     streamed in by TMA
      O_i : [B_r × d]     accumulator, FP32, lives across the
                          whole inner loop

  ══════════════════════════════════════════════════════════════════
  THE STRUCTURAL DIFFICULTY

  GEMM1's A-operand is GEMM0's output, transformed by softmax.

  That is a genuine serial dependency inside every iteration —
  you cannot start PV for block j until QK for block j has
  finished AND its softmax has been applied.

  Compare a plain GEMM (Chapter 9): its inner loop is a pure
  chain of independent wgmma calls with nothing in between.
  Attention has a 256×-slower operation wedged in the middle
  of every iteration.

  Techniques 2 and 3 are two different answers to the same
  question: what else can the tensor cores be doing while
  that exp() runs?
  ══════════════════════════════════════════════════════════════════

  Typical tile shapes (BF16, d = 128, H100):

    B_r (Q rows per block)    64 or 128
    B_c (K/V rows per block)  64 … 176
    d                         64, 128, 256

    SMEM budget, B_r=128, B_c=128, d=128, BF16, 2 stages:
      Q tile:  128 × 128 × 2                 =  32 KB  (resident)
      K tile:  128 × 128 × 2 × 2 stages      =  64 KB
      V tile:  128 × 128 × 2 × 2 stages      =  64 KB
      ────────────────────────────────────────────────
                                               160 KB
      Against Chapter 1's 228 KB limit — comfortable, and the
      headroom is what lets you raise the stage count.
```

---

## 5 — Technique 1 — Producer/Consumer Warp Specialization

The first and largest win, and if you have read Chapter 8 §4 there is nothing
conceptually new here — which is exactly the point. Most of FA3's advantage over
FA2 comes from applying textbook Hopper kernel design to a kernel that had none.

### 5.1 The Warpgroup Split

```
FA3 Block Layout — Who Does What
══════════════════════════════════════════════════════════════════════

  One thread block = 384 threads = 12 warps = 3 warpgroups

  ┌────────────────────────────────────────────────────────────────┐
  │  WARPGROUP 0  (warps 0-3, 128 threads)   PRODUCER              │
  │  ────────────────────────────────────────────────────────────  │
  │  Only ONE thread is really busy. It:                           │
  │    • issues cp.async.bulk.tensor for K_j and V_j               │
  │    • does mbarrier.arrive.expect_tx for the byte count          │
  │    • waits on the "buffer free" barrier before reusing a slot  │
  │  Register budget: setmaxnreg.dec  →  32 registers              │
  ├────────────────────────────────────────────────────────────────┤
  │  WARPGROUP 1  (warps 4-7)                CONSUMER A            │
  │  WARPGROUP 2  (warps 8-11)               CONSUMER B            │
  │  ────────────────────────────────────────────────────────────  │
  │  Each independently processes its own set of Q rows:           │
  │    • wgmma QK^T                                                │
  │    • online softmax on the result                              │
  │    • wgmma PV, accumulating into O                             │
  │  Register budget: setmaxnreg.inc  →  160-232 registers         │
  └────────────────────────────────────────────────────────────────┘

  Why two consumers rather than one?
    §6. Two consumers is what makes ping-pong possible.
    With one consumer there is nobody to run GEMMs while it
    does softmax.

  Why the register asymmetry?
    Chapter 8 §5. The producer holds a descriptor and a loop
    counter. The consumer holds a [B_r × d] FP32 accumulator —
    for B_r=128, d=128 that is 64 registers of O alone, plus
    S, plus P, plus the softmax statistics. Every register the
    producer gives up is one the consumer can use to avoid
    spilling.
```

### 5.2 The Producer Loop

```cuda
// ─────────────────────────────────────────────────────────────────
// REFERENCE — illustrates structure; not compiled or benchmarked.
// Producer warpgroup: fetch K_j and V_j, forever, into a circular
// buffer of NUM_STAGES slots. See Chapter 3 §12 for the barrier
// reuse pattern and Chapter 5 §7 for the completion mechanism.
// ─────────────────────────────────────────────────────────────────

__device__ void producer_loop(
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    SharedStorage&  smem,
    int             k_block_begin,
    int             k_block_end,
    int             q_block_idx)
{
    // Give registers back to the consumers (Chapter 8 §5).
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(32));

    const int lane = threadIdx.x % 32;
    const int warp = (threadIdx.x / 32) % 4;

    // One warp, one thread issues everything. The rest exit.
    if (warp != 0 || lane != 0) return;

    // Bytes the TMA will deposit per stage: one K tile + one V tile.
    constexpr uint32_t kBytesPerStage =
        (kBlockN * kHeadDim + kBlockN * kHeadDim) * sizeof(Element);

    int  stage = 0;
    uint32_t phase = 0;          // Chapter 3 §10.1 — the reuse bit

    for (int j = k_block_begin; j < k_block_end; ++j) {

        // 1. Wait until the consumers have finished with this slot.
        //    This is the WAR half of the two-barrier handshake
        //    (Chapter 9 §9): "buffer free" must be signalled before
        //    we are allowed to overwrite it.
        mbarrier_wait(&smem.buffer_free[stage], phase);

        // 2. Declare how many bytes are about to arrive, and arrive
        //    ourselves. Chapter 3 §9.2 — expect_tx does both jobs.
        mbarrier_arrive_expect_tx(&smem.data_ready[stage],
                                  kBytesPerStage);

        // 3. Fire the two copies. Both return immediately; the TMA
        //    will decrement data_ready[stage] as bytes land.
        tma_load_2d(tma_k, smem.k[stage], &smem.data_ready[stage],
                    /*coord=*/{0, j * kBlockN});
        tma_load_2d(tma_v, smem.v[stage], &smem.data_ready[stage],
                    /*coord=*/{0, j * kBlockN});

        // 4. Advance. The phase bit flips each time we wrap.
        if (++stage == kNumStages) { stage = 0; phase ^= 1; }
    }
}
```

Three things to notice, all of them Chapter 3 and 5 material:

- **One thread issues the whole tile.** That is the TMA's entire reason for
  existing (Chapter 5 §2). The other 127 threads of the producer warpgroup return
  immediately and cost nothing.
- **`expect_tx` before the copy, never after.** Chapter 3 §9.2. Issue the copy
  first and the TMA may complete before the barrier knows how many bytes to
  expect, which hangs the kernel.
- **The `buffer_free` wait is not optional.** Without it you have the WAR hazard
  from Chapter 3 §6.2: the producer overwrites a tile a consumer is still
  reading.

### 5.3 The Consumer Skeleton

```cuda
// ─────────────────────────────────────────────────────────────────
// REFERENCE — illustrates structure; not compiled or benchmarked.
// Consumer warpgroup. This version is the SIMPLE one: GEMM, then
// softmax, then GEMM, strictly in order. §6 and §7 rearrange it.
// ─────────────────────────────────────────────────────────────────

__device__ void consumer_loop_naive(
    SharedStorage& smem,
    float4         (&O)[kOAccumRegs],   // [B_r × d] accumulator
    float          (&m)[kRowsPerThread], // running max   (§2)
    float          (&l)[kRowsPerThread], // running sum   (§2)
    int            k_block_begin,
    int            k_block_end)
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(160));

    int  stage = 0;
    uint32_t phase = 0;

    for (int j = k_block_begin; j < k_block_end; ++j) {

        // Wait for K_j and V_j to have landed.
        mbarrier_wait(&smem.data_ready[stage], phase);

        // ── GEMM0: S = Q · K_j^T ──────────────────────────────
        float S[kSAccumRegs];
        wgmma_fence();                       // Chapter 6 §4
        wgmma_qk(S, smem.q, smem.k[stage]);  // wgmma.mma_async
        wgmma_commit_group();
        wgmma_wait_group<0>();               // ◄── FULL STALL

        // ── SOFTMAX (§2) ──────────────────────────────────────
        //     Tensor cores are IDLE for this entire block.
        float alpha[kRowsPerThread];
        online_softmax_update(S, m, l, alpha);   // exp() on SFUs
        rescale_accumulator(O, alpha);           // O *= alpha

        // ── GEMM1: O += P · V_j ───────────────────────────────
        wgmma_fence();
        wgmma_pv(O, S /* now P */, smem.v[stage]);
        wgmma_commit_group();
        wgmma_wait_group<0>();               // ◄── FULL STALL

        // Release the slot back to the producer.
        mbarrier_arrive(&smem.buffer_free[stage]);

        if (++stage == kNumStages) { stage = 0; phase ^= 1; }
    }
}
```

This already gets you to ~570 TFLOP/s, because the *loads* now overlap perfectly
with compute. But look at the two `wgmma_wait_group<0>()` calls and the block
between them. The tensor cores stop dead during softmax, exactly as the arithmetic
in §3.2 predicted. That is the remaining 42%.

---

## 6 — Technique 2 — Ping-Pong Scheduling

### 6.1 The Idea

If one warpgroup must idle its tensor cores during softmax, give the tensor cores
a *different* warpgroup to serve.

```
Ping-Pong — Staggering Two Consumers
══════════════════════════════════════════════════════════════════════

  WITHOUT ping-pong (both warpgroups in lockstep):

  time ──────────────────────────────────────────────────────────►
  WG1   [GEMM0][softmax][GEMM1] [GEMM0][softmax][GEMM1]
  WG2   [GEMM0][softmax][GEMM1] [GEMM0][softmax][GEMM1]
        ▲▲▲▲▲▲        ▲▲▲▲▲▲▲
        both compute   BOTH doing softmax — tensor cores idle

  Tensor core utilisation: ~66%
  (both warpgroups leave the same hole at the same time)


  WITH ping-pong (WG2 offset by half a phase):

  time ──────────────────────────────────────────────────────────►
  WG1   [GEMM0][softmax][GEMM1][GEMM0][softmax][GEMM1]
  WG2          [GEMM0 ][softmax][GEMM1][GEMM0 ][softmax]
               ▲                ▲
               WG2 computes     WG1 computes
               while WG1        while WG2
               does softmax     does softmax

  Tensor core utilisation: ~85%
  The holes interleave instead of coinciding.

  ══════════════════════════════════════════════════════════════════
  This is exactly the ping-pong architecture from Chapter 9 §14.
  The only difference is what fills the gap: in a plain GEMM it
  is the epilogue, here it is softmax.
  ══════════════════════════════════════════════════════════════════
```

### 6.2 Forcing the Stagger with Named Barriers

The hardware will not stagger the warpgroups for you. Left alone, both consumers
start at the same instant and stay in lockstep forever. You have to impose the
offset, and named barriers (Chapter 3 §16) are how.

```cuda
// ─────────────────────────────────────────────────────────────────
// REFERENCE — illustrates structure; not compiled or benchmarked.
// Two named barriers, used as a baton passed back and forth so that
// WG1's GEMMs are always scheduled ahead of WG2's.
// ─────────────────────────────────────────────────────────────────

// Named barrier IDs. 0 is reserved by __syncthreads().
constexpr int kBarrierWG1 = 1;
constexpr int kBarrierWG2 = 2;

// Both consumer warpgroups participate: 2 × 128 threads.
constexpr int kConsumerThreads = 256;

__device__ inline void named_barrier_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ inline void named_barrier_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ void consumer_loop_pingpong(/* ... */ int wg_id)
{
    // ── PROLOGUE: create the offset ──────────────────────────────
    // WG2 waits once at the very top. From then on the two
    // warpgroups stay half a phase apart for the whole loop.
    if (wg_id == 1) {
        named_barrier_sync(kBarrierWG1, kConsumerThreads);
    }

    for (int j = k_block_begin; j < k_block_end; ++j) {

        // ── GEMM PHASE — one warpgroup at a time ─────────────────
        wgmma_fence();
        wgmma_qk(S, smem.q, smem.k[stage]);
        wgmma_pv(O, P_prev, smem.v[stage_prev]);   // see §7
        wgmma_commit_group();

        // Hand the baton to the other warpgroup: it may now run
        // ITS GEMMs while we go and do softmax.
        named_barrier_arrive(wg_id == 0 ? kBarrierWG2 : kBarrierWG1,
                             kConsumerThreads);

        wgmma_wait_group<0>();

        // ── SOFTMAX PHASE — overlaps the other WG's GEMMs ────────
        online_softmax_update(S, m, l, alpha);
        rescale_accumulator(O, alpha);

        // Wait for our turn at the tensor cores again.
        named_barrier_sync(wg_id == 0 ? kBarrierWG1 : kBarrierWG2,
                           kConsumerThreads);

        if (++stage == kNumStages) { stage = 0; phase ^= 1; }
    }
}
```

```
Why bar.arrive and not bar.sync for the handoff
══════════════════════════════════════════════════════════════════════

  bar.sync  = arrive AND wait      (blocking)
  bar.arrive = arrive, keep going  (non-blocking)

  The handoff must be non-blocking. The whole point is that
  after releasing the other warpgroup, WE continue immediately
  into softmax. Using bar.sync there would park us until the
  other warpgroup caught up — reinstating exactly the lockstep
  we are trying to break.

  This is the same arrive/wait split as Chapter 3 §14.2's
  barrier.cluster, for the same reason: the gap between
  "I'm done" and "everyone's done" is where the useful work goes.
```

### 6.3 Worked Example — Where the Cycles Go

```
Ping-Pong — Cycle Accounting for One Iteration
══════════════════════════════════════════════════════════════════════

  Assumptions (B_r = 128, B_c = 128, d = 128, BF16):

    GEMM0 (QK^T):   2 × 128 × 128 × 128    = 4.19 MFLOP
    GEMM1 (PV):     2 × 128 × 128 × 128    = 4.19 MFLOP
    exp():          128 × 128              = 16,384 ops

  Per-SM rates (Chapter 1 §19):
    tensor BF16:   1024 FLOP/cycle/SM
    SFU:             16 ops/cycle/SM

  Cycle costs:
    GEMM0:   4.19e6 / 1024  =  4,096 cycles
    GEMM1:   4.19e6 / 1024  =  4,096 cycles
    exp:    16,384 / 16     =  1,024 cycles
    (plus ~512 cycles of row reductions, rescaling, bookkeeping)

  ─── SERIAL (one warpgroup, §5.3) ────────────────────────────────
    4096  +  1536  +  4096  =  9,728 cycles per iteration
    Tensor cores busy: 8,192 / 9,728 = 84%
    ... but with TWO warpgroups both stalling at the same time,
    the SM-level figure lands nearer 66%.

  ─── PING-PONG (two warpgroups, offset) ──────────────────────────
    WG1 softmax (1,536 cycles) runs UNDER WG2's GEMM0.
    WG2 softmax runs under WG1's GEMM1.

    Steady-state cost per iteration per warpgroup:
      max(8192 tensor cycles, 8192 + hidden softmax) ≈ 8,192

    Tensor cores busy: ~8,192 / 8,704 = 94% (per warpgroup pair)
    Measured whole-kernel: 570 → 620 TFLOP/s

  ══════════════════════════════════════════════════════════════════
  Note the gap between the model (94%) and reality (63% of peak).
  The remainder is the register pressure that two big accumulators
  create, plus epilogue, plus the tail. Models like this one tell
  you the DIRECTION, never the magnitude — which is precisely why
  Chapter 13 exists.
  ══════════════════════════════════════════════════════════════════
```

---

## 7 — Technique 3 — Intra-Warpgroup 2-Stage Pipelining

Ping-pong hides one warpgroup's softmax behind another's GEMMs. This technique
goes further: hide a warpgroup's softmax behind *its own* GEMMs.

### 7.1 Overlapping Softmax j with GEMM j+1

The trick is a software pipeline of depth two, of exactly the kind Chapter 8 §7
describes — except that the stages are "softmax of iteration j" and "GEMMs of
iteration j+1".

```
2-Stage Intra-Warpgroup Pipeline
══════════════════════════════════════════════════════════════════════

  NAIVE ORDER (§5.3):

    iteration j:   QK_j  →  wait  →  softmax_j  →  PV_j  →  wait
                            ▲▲▲▲                            ▲▲▲▲
                            stall                           stall

  PIPELINED ORDER:

    iteration j:   issue QK_{j+1}      (commit, DO NOT wait)
                   issue PV_j          (commit, DO NOT wait)
                   softmax on S_{j+1}  ◄── computed last iteration,
                                            available NOW
                   wait for the group

  ══════════════════════════════════════════════════════════════════
  The reordering that makes it work:

  At the top of iteration j, S_{j} is ALREADY COMPUTED — it was
  issued during iteration j-1. So we can immediately fire the next
  two GEMMs, and then do this iteration's softmax while they run.

  Rendered as a table, with ● = tensor cores busy:

    iter │ issue QK   │ issue PV   │ softmax    │ tensor busy?
    ─────┼────────────┼────────────┼────────────┼──────────────
      0  │ QK_0       │ —          │ —          │ ●
      1  │ QK_1       │ PV_0       │ softmax_0  │ ● overlapped
      2  │ QK_2       │ PV_1       │ softmax_1  │ ● overlapped
      3  │ QK_3       │ PV_2       │ softmax_2  │ ● overlapped
     ... │            │            │            │
      n  │ —          │ PV_{n-1}   │ softmax_{n-1} │ ●

  Every iteration in the steady state has two wgmma groups in
  flight while the SFUs chew through the previous block's exp().
  ══════════════════════════════════════════════════════════════════
```

```cuda
// ─────────────────────────────────────────────────────────────────
// REFERENCE — illustrates structure; not compiled or benchmarked.
// The 2-stage inner loop. Note that NOTHING waits until after the
// softmax: both wgmma groups are in flight across it.
// ─────────────────────────────────────────────────────────────────

// ── PROLOGUE: prime the pipeline with QK for block 0 ────────────
mbarrier_wait(&smem.data_ready[0], phase);
wgmma_fence();
wgmma_qk(S_cur, smem.q, smem.k[0]);
wgmma_commit_group();
wgmma_wait_group<0>();
online_softmax_update(S_cur, m, l, alpha);   // → P_cur
rescale_accumulator(O, alpha);

// ── STEADY STATE ────────────────────────────────────────────────
for (int j = 1; j < n_blocks; ++j) {

    mbarrier_wait(&smem.data_ready[stage], phase);

    wgmma_fence();

    //  Next iteration's QK — result not needed until next pass.
    wgmma_qk(S_next, smem.q, smem.k[stage]);
    wgmma_commit_group();

    //  THIS iteration's PV, consuming P_cur computed last pass.
    wgmma_pv(O, P_cur, smem.v[stage_prev]);
    wgmma_commit_group();

    //  ── Both groups now in flight. Do softmax underneath them. ──
    //  S_next is not ready yet, so this softmax works on the
    //  scores from the PREVIOUS QK, which completed last iteration.
    wgmma_wait_group<1>();      // ◄── let ONE group stay outstanding
    online_softmax_update(S_next, m, l, alpha);
    rescale_accumulator(O, alpha);
    swap(S_cur, S_next);
    P_cur = S_cur;

    mbarrier_arrive(&smem.buffer_free[stage_prev]);
    stage_prev = stage;
    if (++stage == kNumStages) { stage = 0; phase ^= 1; }
}
// ── EPILOGUE: drain the final PV ────────────────────────────────
wgmma_wait_group<0>();
```

The `wgmma_wait_group<1>()` is the crux. Chapter 7 §3 made the point that the
argument is how many groups may remain *outstanding* — so `<1>` waits for the
older group (the QK whose scores we are about to use) while deliberately leaving
the newer one running. Write `<0>` there and the whole technique evaporates.

### 7.2 The Register Cost

Nothing is free, and this one is paid for in the scarcest resource on the chip.

```
2-Stage Pipelining — The Register Bill
══════════════════════════════════════════════════════════════════════

  Without the pipeline, per consumer thread:
    O accumulator  [B_r × d] / 128 threads, FP32   =  64 regs
    S accumulator  [B_r × B_c] / 128 threads, FP32 =  64 regs
    m, l, alpha    row statistics                   ≈   6 regs
    addresses, descriptors, loop state              ≈  16 regs
                                                     ────────
                                                      150 regs

  With the pipeline you need TWO score buffers live at once
  (S_cur being consumed, S_next being produced):
                                                      +64 regs
                                                     ────────
                                                      214 regs

  Against the 255-per-thread hardware ceiling (Chapter 1 §14),
  that is tight — and it is exactly why the producer warpgroup
  runs setmaxnreg.dec down to 32. Those registers are not being
  saved, they are being TRANSFERRED.

  When it does not fit:
    ├── Reduce B_c (fewer score columns → smaller S)
    ├── Keep S in FP32 but P in BF16 (halves the second buffer)
    └── Fall back to §6 alone — ping-pong without the intra-WG
        pipeline still gets you 620 TFLOP/s

  Chapter 13 §7 shows how to tell spilling from a profile:
    watch for local memory traffic in Memory Workload Analysis,
    and "Stall Long Scoreboard" on instructions that should
    only be touching registers.
```

---

## 8 — Causal Masking and Load Imbalance

Everything so far assumed every query attends to every key. Language models do
not work that way: token *i* may only attend to tokens ≤ *i*. That single
constraint halves the FLOPs and creates a scheduling problem considerably nastier
than the one it solves.

### 8.1 Three Kinds of Block

```
Causal Masking — The Block Taxonomy
══════════════════════════════════════════════════════════════════════

  Q blocks (rows) × K blocks (columns), causal mask:

            K0    K1    K2    K3    K4    K5
         ┌─────┬─────┬─────┬─────┬─────┬─────┐
     Q0  │ DIAG│ SKIP│ SKIP│ SKIP│ SKIP│ SKIP│
         ├─────┼─────┼─────┼─────┼─────┼─────┤
     Q1  │ FULL│ DIAG│ SKIP│ SKIP│ SKIP│ SKIP│
         ├─────┼─────┼─────┼─────┼─────┼─────┤
     Q2  │ FULL│ FULL│ DIAG│ SKIP│ SKIP│ SKIP│
         ├─────┼─────┼─────┼─────┼─────┼─────┤
     Q3  │ FULL│ FULL│ FULL│ DIAG│ SKIP│ SKIP│
         ├─────┼─────┼─────┼─────┼─────┼─────┤
     Q4  │ FULL│ FULL│ FULL│ FULL│ DIAG│ SKIP│
         ├─────┼─────┼─────┼─────┼─────┼─────┤
     Q5  │ FULL│ FULL│ FULL│ FULL│ FULL│ DIAG│
         └─────┴─────┴─────┴─────┴─────┴─────┘

  SKIP  — entirely above the diagonal. Do not launch. Do not
          fetch K or V. Free.

  FULL  — entirely below. No masking needed at all; run the
          plain kernel from §7 with no mask instructions.

  DIAG  — straddles the diagonal. Needs a per-element mask:
          set S[r][c] = -inf where (q_base+r) < (k_base+c).
          Costs a comparison and a select per element.

  ══════════════════════════════════════════════════════════════════
  THE OPTIMISATION THAT MATTERS

  Compile TWO versions of the inner loop and branch once, not
  per element:

    for j in 0 .. diag_block-1:   inner_loop_no_mask()   ◄── FULL
    for j == diag_block:          inner_loop_masked()    ◄── DIAG
    (j > diag_block: never executed)                     ◄── SKIP

  A mask check inside the FULL blocks would cost a predicate
  per element across ~99% of the work, to catch a case that
  cannot occur there. Specialising the loop removes it entirely.
  ══════════════════════════════════════════════════════════════════
```

### 8.2 Balancing the Triangle

```
The Load Imbalance
══════════════════════════════════════════════════════════════════════

  Work per Q block, with 6 K blocks:

    Q0:  1 block   ██
    Q1:  2 blocks  ████
    Q2:  3 blocks  ██████
    Q3:  4 blocks  ████████
    Q4:  5 blocks  ██████████
    Q5:  6 blocks  ████████████

  Ratio between lightest and heaviest: 6×.
  For a 128-block sequence: 128×.

  Naive mapping (one Q block per thread block) on 132 SMs:
    The SM that drew Q0 finishes almost immediately and then
    sits idle for the entire remaining runtime of the kernel.
    Utilisation ≈ 50% — you are paying for the full square
    while doing half the work.

  ─── FIX 1: Pair the blocks ──────────────────────────────────────
    Assign Q_i and Q_{n-1-i} to the same thread block.
      Q0 + Q5:  1 + 6 = 7 units
      Q1 + Q4:  2 + 5 = 7 units
      Q2 + Q3:  3 + 4 = 7 units
    Perfectly balanced, and it costs one line of index arithmetic.

  ─── FIX 2: Persistent kernel + dynamic work queue ───────────────
    Chapter 9 §17. Launch exactly 132 blocks — one per SM — and
    have each pull the next (Q block, K range) pair from an
    atomic counter until the queue drains. Handles ragged batches
    and variable sequence lengths, which FIX 1 does not.

  ─── FIX 3: Stream-K style splitting ─────────────────────────────
    Chapter 10. Split the K-loop itself across thread blocks and
    combine partial (O, m, l) triples afterwards. Most useful
    when the number of Q blocks is smaller than the SM count —
    e.g. single-sequence decoding, where there may be only one
    Q block in the entire launch.

  ══════════════════════════════════════════════════════════════════
  Note that combining partials in FIX 3 is not a plain sum. Two
  partial results carry different running maxima, so merging them
  needs the same rescaling as §2:

    m = max(m_A, m_B)
    O = exp(m_A - m)·O_A + exp(m_B - m)·O_B
    l = exp(m_A - m)·l_A + exp(m_B - m)·l_B

  The online softmax recurrence is associative. That is what
  makes split-K attention possible at all.
  ══════════════════════════════════════════════════════════════════
```

---

## 9 — FP8 FlashAttention

FP8 doubles tensor core throughput, which on paper takes attention from 740
TFLOP/s to something near 1.5 PFLOP/s. In practice FA3 reaches about 1.2 PFLOP/s,
and getting there requires solving three problems that have nothing to do with
speed: accuracy, layout, and outliers.

Chapter 7 §9 made the case that FP8 is a discipline rather than a data type. This
section is what that discipline looks like in a real kernel.

### 9.1 Block Quantization

```
Quantization Granularity
══════════════════════════════════════════════════════════════════════

  FP8 E4M3 maxes out at 448 (Chapter 7 §10). Attention scores and
  activations do not politely stay under 448, so everything must be
  scaled before conversion and unscaled afterwards.

  The question is: one scale factor for what?

  ─── PER-TENSOR (the naive choice) ───────────────────────────────
    One scalar for the whole of Q. One for K. One for V.
      scale_Q = 448 / max(|Q|)

    Cheap. One value to compute, one to store, one multiply.
    And badly behaved: a single outlier anywhere in a
    [8192 × 128] tensor drags the scale down, and every one of
    the other million values loses precision to accommodate it.

  ─── PER-BLOCK (what FA3 does) ───────────────────────────────────
    One scalar per tile — B_r × d for Q, B_c × d for K and V.

      Q: [8192 × 128] with B_r = 128  →  64 scale factors
      K: [8192 × 128] with B_c = 128  →  64 scale factors

    An outlier now only degrades the 128 rows it shares a block
    with, instead of all 8192. Storage overhead: 64 floats
    against 1 MB of data — call it nothing.

    Better still, the tile is exactly the granularity the TMA
    already fetches, so the scale for a tile is a single scalar
    the consumer loads once per iteration.

  ══════════════════════════════════════════════════════════════════
  FUSING THE QUANTIZATION

  The conversion is not a separate kernel. It rides along with
  whatever memory-bound operation immediately precedes attention
  — in a transformer that is usually rotary position embedding.

  RoPE is memory-bound (Chapter 8 §3): it reads Q and K, does a
  handful of FLOPs per element, and writes them back. It is
  waiting on HBM the entire time. Folding a multiply and a
  cvt.satfinite into it costs literally zero additional time.

    read Q → apply RoPE → × scale → cvt.satfinite.e4m3x4 → write

  Chapter 7 §12 has the conversion instruction itself.
  ══════════════════════════════════════════════════════════════════
```

### 9.2 The Layout Problem — Why V Must Be Transposed

This is the FP8 complication that surprises people, and it comes straight out of
Chapter 7 §13.2.

```
The K-Major Constraint
══════════════════════════════════════════════════════════════════════

  Chapter 7 §13.2: FP8 wgmma requires both operands K-MAJOR —
  contiguous along the reduction dimension. BF16 wgmma is
  relaxed about this. FP8 is not.

  Now look at attention's two GEMMs and ask what "K" (the
  reduction dimension) is in each:

    GEMM0:  S = Q · K^T
            reduction is over d (head dim)
            Q is [B_r × d], contiguous in d          ✓ K-major
            K is [B_c × d], contiguous in d          ✓ K-major
            → no problem

    GEMM1:  O = P · V
            reduction is over B_c (the key positions!)
            P is [B_r × B_c], contiguous in B_c      ✓ K-major
            V is [B_c × d],   contiguous in d        ✗ WRONG
            → V must be contiguous in B_c, i.e. TRANSPOSED

  ══════════════════════════════════════════════════════════════════
  So V arrives from HBM in the layout the model stores it in,
  and GEMM1 needs it the other way round. Three options:

  1. Transpose V in HBM before the kernel.
     Costs a full read+write of V — 2× the traffic on a tensor
     you were reading exactly once. Defeats the purpose.

  2. Have the TMA transpose it.
     It cannot. TMA rearranges within a tile (swizzling,
     interleaving — Chapter 4 §10, §11) but does not transpose.

  3. Transpose in SMEM, in-kernel, while other work proceeds.
     ← what FA3 does. See §9.3.
  ══════════════════════════════════════════════════════════════════
```

### 9.3 In-Kernel Transpose with LDSM and STSM

```
SMEM Transpose via ldmatrix / stmatrix
══════════════════════════════════════════════════════════════════════

  The tools are already familiar from Chapter 6 §9 and Chapter 7 §6:

    ldmatrix (LDSM)  SMEM → registers, 8×8 tiles, hardware
                     scatters elements to the right threads
    stmatrix (STSM)  registers → SMEM, the exact inverse

  Both move 128 bytes per warp per instruction and are
  bank-conflict-free by construction.

  The transpose:

    ┌──────────┐   ldmatrix.trans   ┌──────────┐   stmatrix   ┌──────────┐
    │  V tile  │ ─────────────────► │ registers│ ───────────► │ V^T tile │
    │  in SMEM │   (.trans flag     │ (per-    │              │  in SMEM │
    │ [B_c × d]│    transposes the  │  thread) │              │ [d × B_c]│
    └──────────┘    8×8 sub-tiles)  └──────────┘              └──────────┘

  ldmatrix has a .trans qualifier that transposes each 8×8 tile
  as it loads. Composing that with the right addressing pattern
  transposes the whole tile.

  FP8 has one further wrinkle: ldmatrix.trans operates on 16-bit
  elements. For 8-bit data the byte order within each pair comes
  out wrong, so the register contents need a fix-up with the byte
  permute instruction (prmt.b32) before stmatrix writes them back.

  ══════════════════════════════════════════════════════════════════
  THE COST, AND WHY IT IS ACCEPTABLE

  Per V tile [128 × 128] FP8:
    16 KB of data
    ~64 ldmatrix + ~64 prmt + ~64 stmatrix instructions per warp
    a few hundred cycles

  Against the ~8,000 tensor-core cycles per iteration from §6.3,
  that is under 5% — AND it can be overlapped. The transpose of
  V_{j+1} happens during the GEMMs of iteration j, using the
  same pipelining logic as §7.

  This is warp specialization again: FA3's FP8 path dedicates
  additional warps to transposing while the consumers compute.
  ══════════════════════════════════════════════════════════════════
```

### 9.4 Incoherent Processing — The Hadamard Trick

The accuracy problem, and the most elegant idea in the paper.

```
Incoherent Processing
══════════════════════════════════════════════════════════════════════

  THE PROBLEM

  Transformer activations have outliers — a few dimensions whose
  magnitudes are 10-100× the rest. This is well documented and
  appears to be intrinsic to how these models train.

  Quantization hates outliers, because the scale must accommodate
  the largest value:

    values:  [0.1, 0.2, 0.15, 0.3, 45.0]     ◄── one outlier
    scale:   448 / 45.0 = 9.96
    scaled:  [1.0, 2.0, 1.5, 3.0, 448.0]
    to E4M3: [1.0, 2.0, 1.5, 3.0, 448.0]

    E4M3 near 1.0 has ~3 decimal digits of resolution, so the
    small values survive — but only just. Push the outlier to
    450× and they quantize to a handful of distinct levels.

  ─────────────────────────────────────────────────────────────────
  THE TRICK

  Pick a random orthogonal matrix M. Because M is orthogonal,
  M·Mᵀ = I, and therefore:

      (Q M)(K M)ᵀ  =  Q M Mᵀ Kᵀ  =  Q Kᵀ

  The scores are UNCHANGED. But each entry of QM is a random
  weighted sum of many entries of Q — so an outlier gets smeared
  across the whole row instead of sitting in one place.

    before:  [0.1, 0.2, 0.15, 0.3, 45.0]   max/mean ≈ 200
    after:   [9.2, -7.1, 11.4, -8.8, 6.3]  max/mean ≈ 1.4

  Same dot products. Far friendlier distribution.

  ─────────────────────────────────────────────────────────────────
  WHY IT IS CHEAP

  M is not a general matrix — it is a random Hadamard transform:
  a Hadamard matrix with random ±1 signs on the diagonal.

    Hadamard transform cost:  O(d log d), not O(d²)
    For d = 128:  128 × 7 = 896 ops vs 16,384 for a dense matmul
    And it is all adds and subtracts — no multiplies at all.

  Like the quantization in §9.1, it fuses into the preceding
  memory-bound operation and costs nothing measurable.

  ─────────────────────────────────────────────────────────────────
  RESULT (from the paper)

    FP8 attention, per-tensor quantization:      baseline RMSE
    FP8 attention, block quant + incoherent:     2.6× lower error

  2.6× is the difference between FP8 attention being a research
  curiosity and being deployable.
  ══════════════════════════════════════════════════════════════════
```

---

## 10 — Putting It All Together

The complete forward kernel, in skeleton. Every piece below has appeared
somewhere in Chapters 3–9; what is new is only the arrangement.

```cuda
// ─────────────────────────────────────────────────────────────────
// REFERENCE IMPLEMENTATION — structure only.
// Not compiled, not benchmarked, not numerically validated.
// Read alongside flash-attention/hopper/flash_fwd_kernel.h.
// ─────────────────────────────────────────────────────────────────

template <int kBlockM,     // B_r — Q rows per block
          int kBlockN,     // B_c — K/V rows per block
          int kHeadDim,    // d
          int kNumStages,  // circular buffer depth (Chapter 8 §8)
          bool kIsCausal>
__global__ __launch_bounds__(384, 1)
void flash_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    FlashParams params)
{
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);
    //  SharedStorage layout — Chapter 5 §12 for the alignment rules:
    //    alignas(128) Element q[kBlockM][kHeadDim];
    //    alignas(128) Element k[kNumStages][kBlockN][kHeadDim];
    //    alignas(128) Element v[kNumStages][kBlockN][kHeadDim];
    //    alignas(8)   uint64_t data_ready[kNumStages];
    //    alignas(8)   uint64_t buffer_free[kNumStages];

    const int warpgroup = threadIdx.x / 128;   // 0 = producer, 1-2 = consumers
    const int q_block   = blockIdx.x;
    const int head      = blockIdx.y;

    // ── Barrier init: exactly one thread, then a block-wide sync ──
    if (threadIdx.x == 0) {
        for (int s = 0; s < kNumStages; ++s) {
            mbarrier_init(&smem.data_ready[s],  1);    // TMA signals
            mbarrier_init(&smem.buffer_free[s], 256);  // 2 consumer WGs
        }
        // Prime buffer_free so the producer may fill every slot once.
        for (int s = 0; s < kNumStages; ++s)
            mbarrier_arrive_n(&smem.buffer_free[s], 256);
    }
    __syncthreads();

    // ── Causal bounds (§8.1): compute the K range, skip the rest ──
    int k_begin = 0;
    int k_end   = params.n_kv_blocks;
    if constexpr (kIsCausal) {
        k_end = min(k_end, (q_block + 1) * kBlockM / kBlockN + 1);
    }

    // ── Q is loaded once and stays resident for the whole loop ────
    if (threadIdx.x == 0)
        tma_load_2d(tma_q, smem.q, &smem.q_ready, {0, q_block * kBlockM});

    // ── Role dispatch ─────────────────────────────────────────────
    if (warpgroup == 0) {
        producer_loop(tma_k, tma_v, smem, k_begin, k_end, q_block);   // §5.2
    } else {
        mbarrier_wait(&smem.q_ready, 0);

        float O[kOAccumRegs];                 // [B_r × d], FP32
        float m[kRowsPerThread];              // running max  (§2)
        float l[kRowsPerThread];              // running sum  (§2)
        init_accumulators(O, m, l);

        consumer_loop_pingpong_pipelined(     // §6 + §7 combined
            smem, O, m, l, k_begin, k_end,
            /*wg_id=*/warpgroup - 1,
            /*causal=*/kIsCausal);

        // ── Epilogue ──────────────────────────────────────────────
        //  1. Finalize: O /= l                     (§2.2 step 6)
        //  2. Convert FP32 accumulator → BF16      (Chapter 7 §8 —
        //     stmatrix has no 32-bit form)
        //  3. stmatrix into SMEM                   (Chapter 7 §6)
        //  4. TMA store SMEM → HBM                 (Chapter 5 §10.2)
        //  5. Optionally write logsumexp = m + log(l) for the
        //     backward pass
        finalize_and_store(O, m, l, smem, params, q_block, head);
    }
}
```

```
Launch Configuration
══════════════════════════════════════════════════════════════════════

  dim3 block(384);                     // 3 warpgroups (§5.1)
  dim3 grid(n_q_blocks, n_heads, batch);

  Dynamic SMEM (Chapter 5 §12):
    kBlockM=128, kBlockN=128, kHeadDim=128, kNumStages=2, BF16

      Q:            128 × 128 × 2                    =  32 KB
      K:      2 ×   128 × 128 × 2                    =  64 KB
      V:      2 ×   128 × 128 × 2                    =  64 KB
      barriers:     ~2 × 2 × 8 B                     ≈   0 KB
                                                      ───────
                                                       160 KB

    Requires opting in beyond the 48 KB static limit:

      cudaFuncSetAttribute(flash_fwd_kernel<...>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024);

  Occupancy: 1 block/SM. That is intentional, not a failure —
  Chapter 8 §6. One block of 384 threads with big accumulators
  and a deep pipeline beats two blocks fighting over registers.
```

---

## 11 — Measured Results

Numbers from the FlashAttention-3 paper, H100 SXM5.

```
FA3 vs FA2 — Forward Pass, BF16
══════════════════════════════════════════════════════════════════════

  Technique ladder (from §4.1):

    FlashAttention-2 on H100              ~350 TFLOP/s    35%
    + warp specialization + TMA           ~570 TFLOP/s    58%
    + ping-pong                           ~620 TFLOP/s    63%
    + intra-warpgroup 2-stage             ~660 TFLOP/s    67%
    + tuning                          up to 740 TFLOP/s   75%

  Speedup over FA2:   1.5 – 2.0×  (forward)
                      1.5 – 1.75× (backward)

  FP8 forward:        ~1.2 PFLOP/s
  FP8 accuracy:       2.6× lower error than per-tensor
                      quantization, with block quant +
                      incoherent processing (§9.4)

  Behaviour across shapes:
    ├── Head dim 64, 128, 256 all benefit; 128 is the sweet spot
    ├── Sequence 512 → 16k; longer sequences amortise the
    │   prologue better and land nearer the top of the range
    └── Short sequences are dominated by launch and epilogue
        overhead — the techniques here need a long inner loop
        to pay off
```

```
What the Remaining 25% Is
══════════════════════════════════════════════════════════════════════

  75% of peak is excellent. It is not 100%, and it is worth
  knowing where the rest goes — these are the same categories
  Chapter 13 teaches you to measure.

  ├── Softmax that ping-pong and pipelining cannot hide.
  │   The overlap is good, not perfect; row reductions have
  │   serial dependencies that resist scheduling.
  │
  ├── Prologue and epilogue. The first iteration has nothing
  │   to overlap with; the last has nothing to overlap into.
  │   For 32 K-blocks that is ~6% of iterations running
  │   un-pipelined.
  │
  ├── Register pressure. §7.2 — 214 registers per thread means
  │   one block per SM, so there is no second block to cover
  │   a stall.
  │
  ├── The causal tail (§8). Even with paired scheduling, the
  │   diagonal blocks do masked work at a lower rate.
  │
  └── HBM, occasionally. At d=64 the arithmetic intensity is
      halved and short sequences can slip back toward
      memory-bound. Chapter 8 §1's roofline tells you when.
```

---

## 12 — Quick Reference

### The Online Softmax Recurrence

```
State per row:  m (max, init -inf),  l (sum, init 0),  O (accum, init 0)

Per block S_j:
    m_new = max(m_old, rowmax(S_j))
    α     = exp(m_old - m_new)
    P_j   = exp(S_j - m_new)
    l_new = α·l_old + rowsum(P_j)
    O_new = α·O_old + P_j·V_j

Finalize once:  O = O / l
For backward:   save logsumexp = m + log(l)

Merging two partials (split-K, §8.2):
    m = max(m_A, m_B)
    O = exp(m_A-m)·O_A + exp(m_B-m)·O_B
    l = exp(m_A-m)·l_A + exp(m_B-m)·l_B
```

### The Three Techniques

```
Technique                  Mechanism                        Gain
──────────────────────────────────────────────────────────────────
Warp specialization        Producer WG runs TMA;            350 → 570
  (§5)                     consumer WGs run wgmma;
                           setmaxnreg moves registers

Ping-pong                  Two consumer WGs offset by       570 → 620
  (§6)                     half a phase via named
                           barriers; softmax of one
                           hides under GEMMs of other

2-stage intra-WG           Issue QK_{j+1} and PV_j,         620 → 660
  (§7)                     then softmax_j underneath
                           both; wait_group<1> not <0>

FP8 (§9)                   Block quant + V transpose        740 → 1200
                           + Hadamard incoherent proc.
```

### Warpgroup Roles

```
WG   Warps   Role       setmaxnreg   Responsibility
──────────────────────────────────────────────────────────────────
0    0-3     Producer   dec →  32    TMA loads of K_j, V_j;
                                     expect_tx; buffer_free waits
1    4-7     Consumer   inc → 160+   QK, softmax, PV for its Q rows
2    8-11    Consumer   inc → 160+   same, offset half a phase
```

### Causal Block Types

```
Type   Condition                       Action
──────────────────────────────────────────────────────────────────
SKIP   k_block > q_block               Never launched; K/V not fetched
FULL   k_block < q_block               Unmasked inner loop (no
                                       predicates at all)
DIAG   k_block == q_block              Masked inner loop; S = -inf
                                       where (q_base+r) < (k_base+c)

Balance with: paired Q blocks (i, n-1-i), or a persistent
kernel with a work queue (Chapter 9 §17), or Stream-K
splitting of the K loop (Chapter 10).
```

### FP8 Checklist

```
□ Block quantization, not per-tensor — one scale per tile (§9.1)
□ Fuse the quantize into RoPE or whatever precedes attention
□ E4M3 forward (range 448); E5M2 only if gradients (Chapter 7 §10)
□ V transposed in SMEM before GEMM1 — K-major is mandatory (§9.2)
□ ldmatrix.trans + prmt fix-up for 8-bit elements (§9.3)
□ Hadamard incoherent processing on Q and K (§9.4)
□ K-slice the accumulation — Chapter 7 §14, the FP22 issue
□ Descale the output by (scale_Q · scale_K · scale_V)
```

### Common Mistakes

```
Mistake                                  Symptom            Fix
──────────────────────────────────────────────────────────────────────
wgmma_wait_group<0> in the 2-stage       No speedup from    Use <1>;
loop                                     §7                 Ch 7 §3

Rescaling O before the fence             Corrupt output     wgmma.fence
                                                            before
                                                            touching
                                                            accumulator
                                                            (Ch 7 §5)

Forgetting buffer_free arrive            Hang after         Ch 3 §13 —
                                         kNumStages iters   WAR barrier

expect_tx after issuing the TMA copy     Intermittent hang  Always before
                                                            (Ch 3 §9.2)

Mask predicate inside FULL blocks        ~10% slower        Specialise the
                                                            loop (§8.1)

Per-tensor FP8 scaling                   Accuracy collapse  Block quant
                                         on outlier heads   (§9.1)

V not transposed for FP8 GEMM1           Silently wrong     In-kernel
                                         output             transpose
                                                            (§9.3)

Both warpgroups sync on the same         Ping-pong does     Two barrier
barrier                                  nothing            IDs (§6.2)
```

---

> **Next — Chapter 13: Profiling with Nsight Compute**
>
> This chapter reasoned about performance: cycle counts, arithmetic intensity,
> where the overlap should be. Chapter 13 is about measuring whether any of it is
> true. Its roofline view is Chapter 8 §1 rendered from real counters, its warp
> state breakdown tells you which of the stalls described here is actually
> costing you cycles, and its memory chart maps directly onto Chapter 1's
> hierarchy.
