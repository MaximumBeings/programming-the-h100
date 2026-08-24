# Profiling with Nsight Compute
## Chapter 13 Companion Guide — Measuring What the Rest of This Book Claims

> **Tool versions move.** Nsight Compute ships with the CUDA Toolkit and its
> section names, metric names and CLI flags shift between releases. Every command
> here is written for the 2025.x line (CUDA 12.9 / 13.x). Where a name may have
> changed, the text says which `--list-*` command will tell you the truth on your
> install rather than asserting a version-specific answer.
> **Docs:** [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

**What this covers:** Getting `ncu` to run at all, including the permission error
that stops most people on their first attempt; what replay is and why your kernel
runs several times; the Speed of Light section and how to read it in ten seconds;
the roofline chart as a rendering of Chapter 8 §1 from real counters; Memory
Workload Analysis and the memory chart against Chapter 1's hierarchy; scheduler
and warp state statistics, with every stall reason mapped to the chapter that
explains it; occupancy and why maximising it is the wrong goal; the metric naming
scheme so you can construct queries rather than hunt for them; the SASS source
view; what to look at specifically on Hopper kernels using TMA, `wgmma` and
`mbarrier`; and four worked diagnoses from symptom to fix.

**How to read this chapter.** Every previous chapter reasoned about performance.
This one measures it — and the two disagree more often than anyone would like.

The chapter is deliberately practical. Sections 1–3 get you running and explain
what the tool is doing to your program. Sections 4–8 walk the report sections in
the order you should actually read them, which is not the order the UI presents
them. Sections 9–10 go one level down, to raw metrics and assembly. Sections
11–12 are Hopper-specific and then worked examples.

If you have a kernel in front of you right now, read §2, §4, §7 and then jump
straight to §12. That is the shortest path from "it's slow" to "here is the
line."

---

## ★ BEFORE YOU START — Setup

### What You Need

```
Requirements
══════════════════════════════════════════════════════════════════════

  Nsight Compute ships WITH the CUDA Toolkit. If you have nvcc,
  you almost certainly have ncu:

      ncu --version
      ncu-ui --version         # the GUI, if a display is available

  If missing, it is a standalone download from
  developer.nvidia.com/nsight-compute — and the standalone build
  is often newer than the one bundled with your toolkit, which
  matters because Hopper metric coverage improved across the
  2024-2025 releases.

  Hardware:
    Any CUDA GPU. This chapter's examples assume H100, and §11
    is entirely Hopper-specific.

  Access:
    You need a GPU you can profile — see §2.2. Many shared
    clusters and most cloud instances restrict profiling
    counters by default, and the error message is unhelpful.
```

### The One Thing to Do Before Profiling Anything

```
Compile with line info
══════════════════════════════════════════════════════════════════════

    nvcc -std=c++17 -arch=sm_90a -O2 -lineinfo kernel.cu -o kernel
                                        ▲▲▲▲▲▲▲▲▲

  Without -lineinfo:
    The Source page (§10) shows SASS assembly with no idea which
    line of your CUDA it came from. You can still profile, but
    every finding ends at "some instruction somewhere".

  With -lineinfo:
    Every SASS instruction is attributed to a source line, and
    stall samples land on the statement that caused them.

  It costs nothing at runtime. There is no reason to build a
  kernel you intend to profile without it.

  Do NOT use -G (full device debug). It disables optimisation
  entirely and profiles a program that bears no relationship to
  your release build.
```

---

## Table of Contents

1. [Why Profile at All](#1--why-profile-at-all)
2. [Getting ncu Running](#2--getting-ncu-running)
3. [Replay — What ncu Does to Your Program](#3--replay--what-ncu-does-to-your-program)
4. [Speed of Light — The First Screen](#4--speed-of-light--the-first-screen)
5. [The Roofline Chart](#5--the-roofline-chart)
6. [Memory Workload Analysis](#6--memory-workload-analysis)
7. [Scheduler and Warp State Statistics](#7--scheduler-and-warp-state-statistics)
8. [Occupancy — and Why It Is Not the Goal](#8--occupancy--and-why-it-is-not-the-goal)
9. [The Metric Naming Scheme](#9--the-metric-naming-scheme)
10. [The Source Page — SASS and Stall Sampling](#10--the-source-page--sass-and-stall-sampling)
11. [Profiling Hopper Kernels](#11--profiling-hopper-kernels)
12. [Four Worked Diagnoses](#12--four-worked-diagnoses)
13. [Quick Reference](#13--quick-reference)

---

## 1 — Why Profile at All

You have just spent twelve chapters learning to reason about GPU performance from
first principles. This section argues that the reasoning is necessary and not
sufficient, which is a slightly awkward thing for a book to say about itself.

```
What Reasoning Gets Right, and What It Doesn't
══════════════════════════════════════════════════════════════════════

  MODELS ARE GOOD AT DIRECTION
    ├── "This kernel is memory-bound"           usually correct
    ├── "Coalescing this will help"             usually correct
    ├── "Bank conflicts are costing me"         usually correct
    └── "Overlapping these should be a win"     usually correct

  MODELS ARE BAD AT MAGNITUDE
    ├── "This should run at 80% of peak"        rarely correct
    ├── "This will be 2× faster"                rarely correct
    └── "The bottleneck is X"                   correct maybe half
                                                 the time

  Chapter 12 §6.3 is an honest example. A careful cycle model
  predicted ~94% tensor core utilisation for the ping-pong
  pipeline. Measured: 63% of peak. The model was not wrong about
  the technique — ping-pong genuinely helps — it was wrong about
  everything else that was also happening.

  ══════════════════════════════════════════════════════════════════
  THINGS THAT ROUTINELY SURPRISE PEOPLE, AND WHICH ONLY A
  PROFILER WILL TELL YOU:

  ├── The compiler spilled registers you did not expect
  ├── The kernel is limited by instruction issue, not by any
  │   resource you were thinking about
  ├── A memory access you assumed coalesced does not
  ├── Two-thirds of the runtime is in a prologue you never
  │   considered
  ├── Occupancy is 12% because of one array you forgot about
  └── The kernel you have been optimising is 4% of total runtime
  ══════════════════════════════════════════════════════════════════

  The workflow this chapter argues for:

      reason  →  implement  →  MEASURE  →  reason again

  Not: reason → implement → ship.
```

---

## 2 — Getting ncu Running

### 2.1 The Two Front Ends

```
ncu vs ncu-ui
══════════════════════════════════════════════════════════════════════

  ncu      Command-line. Collects data. Can print a summary
           straight to the terminal, or save a .ncu-rep file.
           This is what you run on a remote H100.

  ncu-ui   Graphical. Opens a .ncu-rep. Charts, source
           correlation, side-by-side baselines.
           This is what you run on your laptop.

  The workflow that actually works for cloud GPUs:

      # on the H100 box
      ncu --set full -o myreport ./my_kernel

      # on your machine
      scp gpu-box:myreport.ncu-rep .
      ncu-ui myreport.ncu-rep

  The report file is self-contained. You do not need the binary,
  the source, or the GPU to open it — though with -lineinfo and
  the sources present, ncu-ui will show your code alongside.
```

### 2.2 The Permission Problem

This stops more first attempts than anything else, and the error text does not
say what to do.

```
ERR_NVGPUCTRPERM
══════════════════════════════════════════════════════════════════════

  ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission
  to access NVIDIA GPU Performance Counters on the target device.

  WHY: since 2019, NVIDIA's driver restricts performance counters
  to administrators by default. The counters can observe activity
  from other processes on the same GPU, so on a shared machine
  they are a genuine information-disclosure channel.

  ─── FIX 1: run as root (quickest, needs sudo) ───────────────────

      sudo ncu --set full -o report ./my_kernel

    Note that sudo may not carry your CUDA environment. If it
    fails to find libraries:
      sudo env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
           ncu --set full -o report ./my_kernel

  ─── FIX 2: permit it system-wide (best on your own box) ─────────

      # persist across reboots
      echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
      sudo update-initramfs -u        # Debian/Ubuntu
      sudo reboot

    Or without a reboot, if nothing is using the GPU:
      sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
      sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

  ─── FIX 3: containers ───────────────────────────────────────────

    Docker needs the capability explicitly:
      docker run --gpus all --cap-add=SYS_ADMIN ...

    Without SYS_ADMIN you get the same error inside a container
    even when the host allows profiling.

  ─── FIX 4: managed cloud / shared cluster ───────────────────────

    Some providers disable this and will not enable it. Options:
      ├── a bare-metal or dedicated instance type
      ├── ask the admin (the modprobe flag is the ask)
      └── fall back to CUPTI-free timing: cudaEvent timings plus
          the reasoning in Chapters 8 and 12. Worse, but not
          nothing.

  ══════════════════════════════════════════════════════════════════
  Official page: developer.nvidia.com/
                 nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute
```

### 2.3 Your First Profile

```
The Minimal Useful Command
══════════════════════════════════════════════════════════════════════

  ── Fastest possible look ────────────────────────────────────────

      ncu ./my_kernel

    Runs with the default set. Prints to the terminal. Profiles
    EVERY kernel launch, which on a real application means you
    will wait a long time and drown in output.

  ── What you actually want ───────────────────────────────────────

      ncu --set full \
          --kernel-name flash_fwd_kernel \
          --launch-skip 5 \
          --launch-count 1 \
          -o report \
          ./my_kernel

    --set full          collect everything, including roofline
    --kernel-name       only this kernel (substring match; use
                        --kernel-name-base demangled for C++)
    --launch-skip 5     ignore the first 5 launches — they hit
                        cold caches and JIT
    --launch-count 1    profile exactly one launch
    -o report           write report.ncu-rep instead of printing

  ── Terminal summary without the GUI ─────────────────────────────

      ncu --set full --launch-count 1 --print-summary per-kernel \
          ./my_kernel

      # or re-read a saved report
      ncu --import report.ncu-rep --page details

  ══════════════════════════════════════════════════════════════════
  ALWAYS SKIP THE FIRST LAUNCHES

  The first launch of a kernel pays for module load, JIT if the
  cubin does not match the GPU, cold instruction cache, cold L2,
  and cold TLB. It can be several times slower than steady state
  and it is not what you are shipping.

  If your program launches the kernel once, wrap it in a loop of
  ten and profile launch 5.
  ══════════════════════════════════════════════════════════════════
```

### 2.4 Sets, Sections, and Metrics

Three levels of granularity, and choosing the right one is mostly about how much
overhead you can tolerate.

```
--set  ⊃  --section  ⊃  --metrics
══════════════════════════════════════════════════════════════════════

  ── SETS: named bundles of sections ──────────────────────────────

      ncu --list-sets

    Typically:
      basic       Speed of Light only. One replay pass. Fast.
      default     SoL + launch + occupancy. A few passes.
      detailed    Most sections. Noticeably slower.
      full        Everything, including roofline. Slowest.

    Rule of thumb: --set full on one launch of one kernel.
    Never --set full on a whole application.

  ── SECTIONS: individual report panels ───────────────────────────

      ncu --list-sections

    The ones that matter, in the order §4-§8 reads them:
      SpeedOfLight                     the 10-second triage
      SpeedOfLight_RooflineChart       §5
      MemoryWorkloadAnalysis           §6
      SchedulerStats                   §7.1
      WarpStateStats                   §7.2
      Occupancy                        §8
      LaunchStats                      grid/block/regs/smem
      ComputeWorkloadAnalysis          pipe utilisation
      InstructionStats                 SASS instruction mix
      SourceCounters                   §10

    Collect a couple:
      ncu --section SpeedOfLight --section WarpStateStats ./app

    Roofline section names have changed across releases and there
    are separate variants for the tensor-core roofline. Check
    --list-sections on YOUR install rather than trusting any
    name printed in a book.

  ── METRICS: individual counters ─────────────────────────────────

      ncu --metrics dram__throughput.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed ./app

    One pass, minimal overhead, exactly the numbers you asked
    for. This is what you use in a tuning loop or a CI check.
    §9 explains how to construct these names.

      ncu --query-metrics                  # everything available
      ncu --query-metrics-mode suffix \
          --metrics dram__throughput       # valid sub-metrics
```

---

## 3 — Replay — What ncu Does to Your Program

Your kernel will run more times than you launched it, and understanding why
explains several otherwise baffling behaviours.

### 3.1 Why Replay Exists

```
The Hardware Counter Shortage
══════════════════════════════════════════════════════════════════════

  An SM has a limited number of physical performance counters —
  far fewer than the number of metrics --set full asks for.

  So ncu collects in PASSES. Run the kernel, collect one group
  of counters. Run it again, collect the next group. Repeat.

  --set full on a Hopper kernel is commonly 10-20 passes.

  ══════════════════════════════════════════════════════════════════
  THE CONSEQUENCE THAT SURPRISES PEOPLE

  Between passes, ncu must restore the kernel's input state —
  otherwise pass 2 runs on the output of pass 1.

  So it saves and restores all memory the kernel writes, around
  every pass. For a kernel touching gigabytes this dominates
  profiling time completely:

    kernel runtime:      2 ms
    × 16 passes:        32 ms
    + save/restore:   ~4000 ms
    ─────────────────────────
    wall clock:       ~4 s for a 2 ms kernel

  This is normal. It is not measuring your kernel as slow —
  the reported Duration is the kernel itself, not the wrapper.
  ══════════════════════════════════════════════════════════════════
```

### 3.2 The Three Replay Modes

```
Replay Modes
══════════════════════════════════════════════════════════════════════

  ── KERNEL REPLAY (default) ──────────────────────────────────────
      --replay-mode kernel

    Replays one kernel at a time, saving/restoring its memory.

    Good:  isolates the kernel; works with any application
    Bad:   serialises concurrent kernels, so anything that
           depends on overlapping streams is measured in a world
           that does not exist
    Bad:   cannot capture host-side interaction mid-kernel

  ── RANGE REPLAY ─────────────────────────────────────────────────
      --replay-mode range

    Replays a whole range of API calls and launches together.
    Ranges come from cuProfilerStart/Stop or NVTX markers.

    Good:  concurrent kernels stay concurrent; multi-kernel
           pipelines are measured honestly
    Bad:   requires the range to be deterministic and
           side-effect-free on the host
    Use:   Chapter 10's dependent-launch protocols. Kernel
           replay actively destroys what you are trying to see.

  ── APPLICATION REPLAY ───────────────────────────────────────────
      --replay-mode application

    Re-runs the ENTIRE application once per pass.

    Good:  no memory save/restore, full program semantics
    Bad:   the application must be deterministic and reasonably
           quick to start
    Use:   kernels whose behaviour depends on host state that
           kernel replay cannot reconstruct

  ══════════════════════════════════════════════════════════════════
  Pick kernel replay unless you have a specific reason. If your
  numbers look strange in a multi-stream program, that reason
  has probably arrived.
```

### 3.3 Clock and Cache Control

```
Two Flags That Change What You Measure
══════════════════════════════════════════════════════════════════════

  ── --clock-control ──────────────────────────────────────────────

    DEFAULT: ncu locks the GPU to base clocks before profiling.

    Why: reproducibility. Boost clocks vary with temperature,
    power and neighbouring work, and a kernel that measures
    1.8 ms now and 2.1 ms in ten minutes tells you nothing about
    your change.

    Consequence: absolute times are PESSIMISTIC. Your kernel in
    production, boosting, will be faster than ncu reports.

      --clock-control none      don't lock; match production

    Use the default while optimising (you want comparable
    numbers). Use `none` when quoting an absolute figure.

  ── --cache-control ──────────────────────────────────────────────

    DEFAULT: ncu flushes all GPU caches between passes.

    Why: reproducibility again. Otherwise pass 1 warms L2 for
    pass 2 and hit rates drift upward across passes.

    Consequence: L2 hit rate is PESSIMISTIC for kernels that in
    reality run back-to-back on data a predecessor left in cache.

      --cache-control none      preserve cache state

    Chapter 4 §12's L2 promotion and Chapter 5 §8's cache hints
    are exactly the features this can hide. If you are measuring
    those, `none` is the honest setting — and re-run several
    times, because now your numbers will move.
```

---

## 4 — Speed of Light — The First Screen

This is the section to read first, always. It answers one question — what class
of problem do I have — in about ten seconds.

### 4.1 The Two Numbers

```
GPU Speed Of Light Throughput
══════════════════════════════════════════════════════════════════════

  Section: SpeedOfLight

  Two headline percentages, each "achieved / theoretical peak":

    Compute (SM) Throughput  [%]     how busy the SM pipes are
    Memory Throughput        [%]     how busy the memory system is

  ══════════════════════════════════════════════════════════════════
  THE TRIAGE TABLE — memorise this

    Compute    Memory     Diagnosis            Go to
    ──────────────────────────────────────────────────────────────
     HIGH       low       Compute bound        §5 roofline,
     (>60%)                                    Ch 8 §2

      low      HIGH       Memory bound         §6 memory analysis,
                (>60%)                         Ch 1 §6, Ch 8 §3

     HIGH      HIGH       Well balanced —      §7 for the last
                          near the ridge       few %

      low       low       LATENCY BOUND        §7 warp states
     (<40%)    (<40%)     ◄── the interesting  ◄── the most
                              case                 common case
    ──────────────────────────────────────────────────────────────

  The bottom row is where most unoptimised kernels land, and it
  is the one people misread. Neither resource is saturated: the
  machine is idle, waiting. Something is stopping warps from
  issuing, and §7 is the section that names it.

  Do NOT respond to low/low by trying to reduce memory traffic
  or FLOPs. Neither is the constraint.
  ══════════════════════════════════════════════════════════════════
```

### 4.2 Also on This Screen

```
The Rest of the Speed of Light Panel
══════════════════════════════════════════════════════════════════════

  Duration                 kernel time. Compare against your own
                           cudaEvent timings — a large gap means
                           you are measuring something else.

  Elapsed Cycles           duration × clock. Useful when you have
                           hand-computed a cycle budget the way
                           Chapter 12 §6.3 does.

  SM Frequency             confirms whether clocks are locked
                           (§3.3). Well below boost = locked.

  DRAM Frequency           HBM clock.

  L1/TEX Cache Throughput  }  where the traffic actually is,
  L2 Cache Throughput      }  before you open §6

  ══════════════════════════════════════════════════════════════════
  Nsight also prints rule-based advice under the tables — lines
  beginning OPT, INF or WRN. These are worth reading and worth
  distrusting. They are static heuristics with no knowledge of
  your algorithm; the "low occupancy" warning in particular fires
  on kernels where low occupancy is a deliberate, correct choice
  (§8, and Chapter 12 §10).
  ══════════════════════════════════════════════════════════════════
```

---

## 5 — The Roofline Chart

Chapter 8 §1 asked you to compute arithmetic intensity by hand and place your
kernel against a ridge point. This section is that exercise performed from real
hardware counters, and it is the single most useful chart in the tool.

### 5.1 Reading It

```
The Roofline
══════════════════════════════════════════════════════════════════════

  Section: SpeedOfLight_RooflineChart  (included in --set full;
  confirm the exact name with --list-sections on your install)

  Log-log axes:
    x = arithmetic intensity   (FLOP per byte)
    y = achieved performance   (FLOP/s)

   FLOP/s
     ▲
     │                    ┌──────────────────────  ← compute ceiling
     │                   ╱                            (peak FLOP/s)
     │                  ╱
     │                 ╱   ● kernel B (compute bound,
     │                ╱      near the ceiling — done)
     │               ╱
     │              ╱
     │        ●    ╱     ← kernel C: BELOW both ceilings.
     │            ╱         Not limited by compute OR memory.
     │           ╱          LATENCY bound. Go to §7.
     │          ╱
     │         ╱ ← memory ceiling, slope = peak bandwidth
     │    ●   ╱
     │       ╱     ● kernel A (memory bound, on the slope —
     │      ╱        improve intensity or coalescing)
     │     ╱
     └────┴────────────────────────────────────────►  FLOP/byte
          ▲
       RIDGE POINT — where the two ceilings meet.
       H100 BF16 tensor: 989e12 / 3.35e12 ≈ 295 FLOP/byte

  ══════════════════════════════════════════════════════════════════
  THREE POSITIONS, THREE ACTIONS

  ON THE DIAGONAL (memory bound)
    You are moving bytes as fast as the machine can. To go
    faster, move fewer bytes: better reuse, better coalescing,
    a fused kernel. Chapter 12 §1.2 is the canonical example —
    fusing attention moved it from 63 to 4095 FLOP/byte.

  ON THE HORIZONTAL (compute bound)
    You are issuing math as fast as the machine can. To go
    faster, do less math, or do it in a cheaper precision
    (Chapter 7 §9), or use a unit with more throughput.

  UNDER BOTH (latency bound)
    The most common and least understood position. Neither
    resource is saturated. Something is stopping issue.
    → §7. Do not touch memory traffic or FLOPs until you
      have read the warp state breakdown.
  ══════════════════════════════════════════════════════════════════
```

### 5.2 The Tensor Core Roofline

```
Why the Default Roofline Can Mislead on Hopper
══════════════════════════════════════════════════════════════════════

  The classic roofline counts FP32 FMA throughput. On an H100
  running wgmma, that is the wrong ceiling by more than an order
  of magnitude:

    FP32 CUDA cores (Ch 1 §12.1):     ~67 TFLOP/s
    BF16 tensor cores (Ch 1 §3):      989 TFLOP/s
    FP8 tensor cores:                1979 TFLOP/s

  Plotted against the FP32 ceiling, a good wgmma kernel appears
  to be far ABOVE the roof — which is nonsense, and a sign you
  are reading the wrong chart.

  Newer Nsight releases include a hierarchical / tensor-core
  roofline with multiple ceilings drawn, one per pipe. Check:

      ncu --list-sections | grep -i roofline

  If it is not available, fall back to reading the pipe
  utilisation directly:

      sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active

  That single number — what fraction of cycles the tensor cores
  were busy — is what you actually wanted from the chart.
  For the Chapter 12 kernel it is the headline metric.
```

### 5.3 Worked Example — Placing a Kernel

```
From Counters to a Point on the Chart
══════════════════════════════════════════════════════════════════════

  Measured for one kernel (numbers illustrative):

    Duration                            1.42 ms
    dram__bytes.sum                    2.85 GB
    sm__sass_thread_inst_executed_op_
      fadd_pred_on.sum  +  fmul  +  2×ffma      = 184 GFLOP

  Compute:

    Achieved FLOP/s   = 184e9  / 1.42e-3 = 129.6 TFLOP/s
    Achieved bytes/s  = 2.85e9 / 1.42e-3 =   2.0 TB/s
    Arithmetic intensity = 184e9 / 2.85e9 = 64.6 FLOP/byte

  Place it:

    Ridge point (FP32): 67e12 / 3.35e12 = 20 FLOP/byte
    64.6 > 20  →  right of the ridge  →  compute-bound region

    Ceiling there is the flat part: 67 TFLOP/s (FP32)
    ... but we measured 129.6 TFLOP/s, which is impossible
    for FP32.

    → The kernel is using tensor cores. Re-plot against the
      BF16 ceiling:
        129.6 / 989 = 13% of tensor peak
        Ridge point (BF16): 989e12/3.35e12 = 295 FLOP/byte
        64.6 < 295  →  MEMORY bound after all

  ══════════════════════════════════════════════════════════════════
  Two lessons, both painful to learn the slow way:

  1. Establish which ceiling applies BEFORE interpreting the
     chart. The same kernel is "compute bound" or "memory bound"
     depending on which one you use.

  2. Achieved throughput above a ceiling is not a triumph, it is
     a diagnostic: you are measuring against the wrong pipe.
  ══════════════════════════════════════════════════════════════════
```

---

## 6 — Memory Workload Analysis

Where Chapter 1's hierarchy diagram meets real traffic counts.

### 6.1 The Memory Chart

```
Reading the Memory Chart
══════════════════════════════════════════════════════════════════════

  Section: MemoryWorkloadAnalysis

  A flow diagram of the hierarchy, annotated with the actual
  bytes and requests that moved. Green boxes are logical units
  (what your instructions asked for), blue are physical
  (what the hardware did).

    ┌──────────┐
    │  Kernel  │
    └────┬─────┘
         │ Global / Local / Shared / Surface / Texture
         ▼
    ┌──────────────────────────┐
    │  L1 / TEX Cache          │  ← hit rate, sectors, wavefronts
    └────┬─────────────────────┘
         │ requests that missed
         ▼
    ┌──────────────────────────┐
    │  L2 Cache                │  ← hit rate, sectors
    └────┬─────────────────────┘
         │ requests that missed
         ▼
    ┌──────────────────────────┐
    │  Device Memory (HBM3)    │  ← bytes, throughput
    └──────────────────────────┘

  ══════════════════════════════════════════════════════════════════
  THE ARITHMETIC TO DO IN YOUR HEAD

  Compare the bytes at each level against what the ALGORITHM
  needs. Chapter 12 §1.2 did exactly this by hand.

  Example: a GEMM reading a 4096×4096 BF16 matrix once needs
  33.5 MB from HBM. If dram__bytes.sum says 260 MB, you are
  re-reading it roughly 8× — your tiling is not achieving the
  reuse you think it is, or L2 is thrashing (Chapter 5 §8).
  ══════════════════════════════════════════════════════════════════
```

### 6.2 Requests, Sectors, Wavefronts

Three words the tool uses that mean different things, and confusing them makes
the numbers unreadable.

```
The Three Units of Memory Traffic
══════════════════════════════════════════════════════════════════════

  REQUEST    One memory instruction from one warp.
             32 threads issuing one ld.global = 1 request.

  SECTOR     A 32-byte chunk of a cache line. This is the
             granularity the cache actually moves (Chapter 1 §6.1).
             A 128-byte line is 4 sectors.

  WAVEFRONT  One cycle of work through the L1 data path. A
             request that cannot be served in one go is split
             into several wavefronts.

  ══════════════════════════════════════════════════════════════════
  THE RATIO THAT MATTERS

      sectors per request  =  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
                              ─────────────────────────────────────────────
                              l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

    32 threads × 4 bytes = 128 bytes = 4 sectors, perfectly
    coalesced. So:

      4      PERFECT      Chapter 1 §6.2's good case
      4-8    fine         some straddling; usually not worth
                          chasing
      8-16   poor         partial coalescing; look at your
                          indexing
      32     WORST CASE   every thread in its own sector.
                          Chapter 1 §12.5's disaster case —
                          32× the bandwidth for the same data

    A ratio of 32 is the single highest-value finding in this
    entire tool. It is usually one transposed index.

  ══════════════════════════════════════════════════════════════════
  SHARED MEMORY BANK CONFLICTS (Chapter 1 §13.3, Chapter 2 §12)

      l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum

    Should be 0 or near it. Anything substantial means threads
    in a warp are hitting the same bank at different addresses.

    Fix: padding or swizzling — Chapter 2 §12.2. If the tile
    came in via TMA, the swizzle mode in the descriptor is
    probably wrong or missing (Chapter 4 §10.4).
```

### 6.3 Cache Hit Rates

```
Hit Rates — and What They Should Be
══════════════════════════════════════════════════════════════════════

    l1tex__t_sector_hit_rate.pct           L1 hit rate
    lts__t_sector_hit_rate.pct             L2 hit rate

  There is no universally good number — it depends entirely on
  your access pattern. What to compare against:

  ├── STREAMING kernel (each byte read once)
  │     L1 ~0%, L2 ~0%. Correct. Caches cannot help. If you
  │     see high hit rates here you are re-reading data.
  │
  ├── TILED GEMM
  │     L2 should be high (60-90%) — tiles are shared between
  │     blocks. L1 lower, since the data lives in shared memory
  │     which bypasses it. A low L2 rate means your tile
  │     scheduling has poor locality — Chapter 9 §18's
  │     rasterisation.
  │
  └── ATTENTION (Chapter 12)
        K and V are re-read by every Q block. High L2 expected.
        If it is low, check the block ordering and consider
        L2 persistence hints (Chapter 5 §8).
```

---

## 7 — Scheduler and Warp State Statistics

The most important sections in the tool, and the ones people skip. If Speed of
Light said low/low, everything you need is here.

### 7.1 Active, Eligible, Issued

```
The Three Warp Counts
══════════════════════════════════════════════════════════════════════

  Section: SchedulerStats

  Per scheduler (Chapter 1 §11 — four per SM), per cycle:

    Active Warps      resident on this scheduler.
                      Ceiling is 16 (Ch 1 §10.1).

    Eligible Warps    ready to issue THIS cycle — operands
                      ready, no hazard, target pipe free.

    Issued Warp       the one actually picked. 0 or 1.

  ══════════════════════════════════════════════════════════════════
  THE DIAGNOSTIC CHAIN

    Active LOW (say < 4)
      → occupancy problem. Not enough warps resident.
        Cause: registers, shared memory, or block size.
        Go to §8.

    Active HIGH, Eligible LOW (say < 1)
      → warps are resident but STALLED. This is the common
        case, and it is what §7.2 is for.

    Eligible HIGH, Issued < 1
      → issue-rate limited. More eligible warps than the
        scheduler can serve. Rare, and a good problem: it means
        you are near the machine's instruction issue ceiling.

  ══════════════════════════════════════════════════════════════════
  A useful rule: "Issue Slot Utilization" below ~30% with high
  active warps means you are latency bound, and the stall
  breakdown below will name the latency.
```

### 7.2 The Stall Reasons

```
Warp Stall Reasons — What Each One Means
══════════════════════════════════════════════════════════════════════

  Section: WarpStateStats
  Metric family: smsp__warp_issue_stalled_<reason>_per_warp_active.pct

  Sorted roughly by how often they are the answer:

  ── Stall Long Scoreboard ────────────────────────────────────────
     Waiting on a global/local memory operation.
     THE most common stall in unoptimised kernels.
     Means: your loads are not being covered by other work.
     Fixes: more warps (§8), better coalescing (§6.2),
            prefetching, or async copy so the wait is explicit
            and overlappable (Chapters 3, 5).
     On Hopper: if this is high in a TMA kernel, your pipeline
            depth is too shallow — Chapter 8 §8.

  ── Stall Short Scoreboard ───────────────────────────────────────
     Waiting on shared memory or a fixed-latency dependency.
     Means: SMEM access is slower than expected.
     Fixes: bank conflicts (§6.2), or too little work between
            dependent SMEM ops.

  ── Stall Barrier ────────────────────────────────────────────────
     Waiting at __syncthreads(), a named barrier, or an
     mbarrier.
     Means: warps arriving at very different times.
     Fixes: load imbalance inside the block (Chapter 12 §8),
            or over-synchronisation — Chapter 3 §16's named
            barriers instead of a whole-block rendezvous.
     On Hopper: high barrier stall in a warp-specialised kernel
            often means producer and consumer are not balanced;
            the producer cannot keep up, or is too far ahead.

  ── Stall MIO Throttle ───────────────────────────────────────────
     The memory input/output pipe is saturated — too many
     shared memory, texture or special-function instructions
     queued.
     Means: often TOO MANY SMEM ops, or heavy SFU use.
     Fixes: vectorise SMEM accesses (ld.shared.v4), or —
            in attention — this is the exp() bottleneck from
            Chapter 12 §3.2 showing up in the profile.

  ── Stall Math Pipe Throttle ─────────────────────────────────────
     The arithmetic pipe is saturated.
     Means: you are compute bound. This is a GOOD stall.
     Fixes: cheaper precision, fewer operations, or nothing.

  ── Stall Wait ───────────────────────────────────────────────────
     Waiting on a fixed-latency instruction dependency
     (an ALU result feeding the next instruction).
     Means: not enough instruction-level parallelism.
     Fixes: unroll, interleave independent chains.

  ── Stall Not Selected ───────────────────────────────────────────
     Was eligible; the scheduler picked someone else.
     Means: NOTHING IS WRONG. This is the machine working.
     High "not selected" with high issue utilisation is the
     picture of a healthy kernel. Do not optimise this.

  ── Stall No Instruction ─────────────────────────────────────────
     Instruction fetch miss — the L0/L1 instruction cache did
     not have the next instruction.
     Means: code footprint too large for L0 (Chapter 1 §15.1).
     Fixes: less aggressive unrolling, less inlining.

  ── Stall Drain / Stall Sleeping ─────────────────────────────────
     Waiting to exit, or explicitly suspended (nanosleep,
     mbarrier try_wait).
     Means: usually benign. In an mbarrier pipeline, sleeping
     is what try_wait is SUPPOSED to do (Chapter 3 §11.1).

  ══════════════════════════════════════════════════════════════════
  READ THE TOP TWO, IGNORE THE TAIL. The distribution is
  usually dominated by one or two reasons; chasing a 3%
  contributor is wasted effort.
```

### 7.3 Stall Reason → Chapter Map

```
Where to Go When You See It
══════════════════════════════════════════════════════════════════════

  Stall reason              Read                    Likely fix
  ──────────────────────────────────────────────────────────────────
  Long Scoreboard           Ch 1 §12.5              coalescing
                            Ch 5 §7                 async copy
                            Ch 8 §8                 deeper pipeline

  Short Scoreboard          Ch 1 §13.3              bank conflicts
                            Ch 2 §12.2              swizzle/pad

  Barrier                   Ch 3 §16                named barriers
                            Ch 8 §4                 warp special.
                            Ch 12 §8                load balance

  MIO Throttle              Ch 1 §12.4              SFU pressure
                            Ch 12 §6                ping-pong

  Math Pipe Throttle        —                       you have won

  Wait                      Ch 8 §7                 pipelining

  No Instruction            Ch 1 §15.1              unroll less

  Not Selected              —                       nothing
```

---

## 8 — Occupancy — and Why It Is Not the Goal

```
Occupancy
══════════════════════════════════════════════════════════════════════

  Section: Occupancy

    Theoretical Occupancy   ceiling from registers, shared memory
                            and block size (Chapter 1 §16.2)
    Achieved Occupancy      what actually happened
      sm__warps_active.avg.pct_of_peak_sustained_active

  The panel also shows which resource is the limiter, which is
  the genuinely useful part.

  ══════════════════════════════════════════════════════════════════
  THE MISCONCEPTION

  Higher occupancy is NOT better. Occupancy is a means of hiding
  latency — Chapter 1 §11.2 — and if you are hiding latency some
  other way, you do not need it.

  Every high-performance Hopper kernel in this book runs at low
  occupancy ON PURPOSE:

    Chapter 9's ping-pong GEMM      1-2 blocks/SM
    Chapter 12's FA3 kernel         1 block/SM, 384 threads
                                    → ~12 of 64 warp slots ≈ 19%

  These kernels hide latency with deep software pipelines and
  async copies, not with a crowd of warps. Registers spent on
  a big accumulator buy more than a second resident block would.

  Nsight's rule engine will warn you about this. It is wrong,
  in this specific case, and it does not know it.

  ══════════════════════════════════════════════════════════════════
  WHEN OCCUPANCY *IS* THE ANSWER

  Achieved ≪ Theoretical
    Something is serialising: tail effect (last wave of blocks
    running alone), imbalance, or divergence. Look at the grid
    size against 132 SMs.

  Theoretical is low AND stalls are Long Scoreboard
    Now you have a real occupancy problem. You are latency bound
    AND have no warps to hide with. Reduce registers
    (__launch_bounds__, Chapter 1 §14.3) or shared memory.

  A useful heuristic: fix occupancy only when the stall
  breakdown (§7.2) says memory latency AND achieved occupancy
  is under ~30%. Otherwise leave it alone.
```

---

## 9 — The Metric Naming Scheme

Once you can construct metric names, you stop searching lists and start asking
direct questions. The scheme looks impenetrable and is actually regular.

```
Metric Name Anatomy
══════════════════════════════════════════════════════════════════════

  unit__(subunit)_(pipestage)_quantity_(qualifiers).rollup.submetric
  ─────  ────────  ──────────  ────────  ──────────  ──────  ────────

  UNIT — which piece of hardware
    sm        streaming multiprocessor
    smsp      SM sub-partition (a quadrant — Chapter 1 §10)
    l1tex     L1 / texture cache
    lts       L2 slice
    ltc       L2 cache
    dram      device memory (HBM3)
    gpu       whole device

  QUANTITY — what is counted
    cycles_active, cycles_elapsed
    bytes, sectors, requests
    inst_executed
    warps_active
    data_bank_conflicts

  ROLLUP — across hardware instances
    .sum   .avg   .min   .max

  SUB-METRIC — how to express it
    .per_second
    .per_cycle_active / .per_cycle_elapsed
    .pct_of_peak_sustained_active     ◄── the useful one
    .pct_of_peak_sustained_elapsed
    .peak_sustained
    .ratio / .pct

  ══════════════════════════════════════════════════════════════════
  ACTIVE vs ELAPSED — the distinction that trips everyone

    ACTIVE   counted only over cycles the unit was doing
             something. "When it was working, how hard?"
    ELAPSED  counted over the whole kernel.
             "Across the whole kernel, how busy?"

    A unit used 10% of the time but saturated when used:
      pct_of_peak_sustained_active  ≈ 100%
      pct_of_peak_sustained_elapsed ≈  10%

    For "is this my bottleneck", you want ELAPSED.
    For "is this unit efficient when used", ACTIVE.
  ══════════════════════════════════════════════════════════════════
```

```
Metrics Worth Memorising
══════════════════════════════════════════════════════════════════════

  ── The triage pair (§4) ─────────────────────────────────────────
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed

  ── Tensor cores (Chapters 6, 7, 12) ─────────────────────────────
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  sm__inst_executed_pipe_tensor.sum

  ── Memory traffic (§6) ──────────────────────────────────────────
  dram__bytes_read.sum
  dram__bytes_write.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_sector_hit_rate.pct
  lts__t_sector_hit_rate.pct

  ── Shared memory (Chapter 1 §13, Chapter 2 §12) ─────────────────
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum

  ── Occupancy (§8) ───────────────────────────────────────────────
  sm__warps_active.avg.pct_of_peak_sustained_active

  ── Stalls (§7.2) ────────────────────────────────────────────────
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  smsp__warp_issue_stalled_barrier_per_warp_active.pct
  smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct

  ── Discovery ────────────────────────────────────────────────────
  ncu --query-metrics                    everything
  ncu --query-metrics-mode suffix \
      --metrics dram__bytes              valid sub-metrics for one
```

---

## 10 — The Source Page — SASS and Stall Sampling

Everything so far has been kernel-level. This is where a finding becomes a line
number.

```
The Source Page
══════════════════════════════════════════════════════════════════════

  Section: SourceCounters
  Requires: -lineinfo at compile time

  Three view modes:
    Source    your CUDA C++, annotated
    SASS      the actual machine instructions
    Source+SASS   side by side, correlated

  ══════════════════════════════════════════════════════════════════
  THE COLUMN THAT MATTERS: Warp Stall Sampling

  The hardware periodically samples what each warp is doing.
  Nsight attributes those samples to the instruction the warp
  was sitting on, so you get a stall histogram PER INSTRUCTION.

    Line   Source                          Stall samples
    ─────────────────────────────────────────────────────────
     42    for (int k = 0; k < K; ++k) {           120
     43        float a = A[row * K + k];        14,820  ◄──
     44        float b = B[k * N + col];           380
     45        acc += a * b;                       210

  Line 43 holds 96% of the stalls. Hover in the UI and it will
  say Long Scoreboard: an uncoalesced global load.

  Look at the indexing: A[row * K + k]. Consecutive threads
  vary `row`, so they are K floats apart — a stride of K, not 1.
  Chapter 1 §12.5's worst case, found in one screen.

  ══════════════════════════════════════════════════════════════════
  READING SASS ON HOPPER — what to look for

  LDGSTS / LDG / STG       global memory ops
  LDSM / STSM              ldmatrix / stmatrix (Ch 6 §9, Ch 7 §6)
  HGMMA / QGMMA            wgmma — BF16/FP16 and FP8 variants
  UBLKCP                   TMA bulk copy (Ch 5)
  BAR.SYNC / BAR.ARV       barriers, named barriers (Ch 3 §16)
  LDL / STL                LOCAL memory — ◄── REGISTER SPILLS

  ══════════════════════════════════════════════════════════════════
  THE FASTEST CHECK IN THE WHOLE TOOL

  Search the SASS for LDL and STL. Any occurrence means the
  compiler spilled registers to local memory, which lives in
  HBM (Chapter 1 §14.3). In a kernel with a large accumulator —
  every kernel in Chapters 9 and 12 — a handful of spills in the
  inner loop can cost more than every other problem combined.

  Faster still, without opening the profiler at all:

      nvcc -arch=sm_90a -O2 --ptxas-options=-v kernel.cu

      ptxas info : Used 232 registers, 0 stack, 0 spill stores,
                   0 spill loads              ◄── what you want
```

---

## 11 — Profiling Hopper Kernels

Generic advice takes you a long way. These are the things specific to the
machine this book is about.

### 11.1 What TMA Looks Like in a Profile

```
Profiling TMA Kernels
══════════════════════════════════════════════════════════════════════

  TMA copies do NOT appear as ordinary load instructions issued
  by your threads. One thread issues one UBLKCP and the engine
  does the rest, so:

  ├── l1tex global load counters stay LOW even though a great
  │   deal of data is moving. This is correct, not a bug.
  ├── dram__bytes still counts the traffic. Use that for
  │   bandwidth, not the L1 counters.
  └── The cost shows up as Stall Barrier on the consumer side
      (waiting on mbarrier), not as Long Scoreboard.

  ══════════════════════════════════════════════════════════════════
  DIAGNOSING A TMA PIPELINE

  Symptom                        Meaning              Fix
  ──────────────────────────────────────────────────────────────────
  High Stall Barrier on          Producer cannot      More stages
  consumer warps                 keep up              (Ch 8 §8);
                                                      check DRAM
                                                      throughput

  High Stall Barrier on          Consumer too slow;   Fewer stages,
  producer warps                 buffers all full     or the
                                                      consumer is
                                                      the real
                                                      bottleneck

  DRAM throughput near peak      Genuinely memory     Improve reuse
  AND barrier stalls high        bound                (Ch 12 §1.2)

  Both low, barrier stalls       Pipeline too         Increase
  high                           shallow              kNumStages
```

### 11.2 Tensor Core Utilisation

```
Is wgmma Actually Running?
══════════════════════════════════════════════════════════════════════

  The single number for a Chapter 6/7/12-style kernel:

      sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active

    > 70%   excellent — FA3 territory
    40-70%  decent; look at what fills the gaps (§7.2)
    < 20%   something is wrong. Check that wgmma is actually
            being issued — grep the SASS (§10) for HGMMA/QGMMA.
            An empty result means your inline PTX was elided or
            you compiled for plain sm_90 instead of sm_90a
            (Chapter 6 §BEFORE YOU START).

  Cross-check against the instruction mix:

      sm__inst_executed_pipe_tensor.sum

  If this is zero, no tensor core instruction executed at all —
  regardless of what your source says.
```

### 11.3 Clusters and Distributed Shared Memory

```
Cluster Kernels (Chapter 2 §1-2)
══════════════════════════════════════════════════════════════════════

  Launch Statistics reports the cluster dimensions, which is
  the quickest confirmation that __cluster_dims__ took effect —
  a surprising number of "clusters didn't help" investigations
  end here, with a cluster size of 1.

  DSMEM traffic appears as shared memory traffic, but crossing
  the SM-to-SM interconnect. Watch for:

  ├── Higher shared memory latency than a local access
  │   (~50-80 cycles vs ~10 — Chapter 2 §7.1)
  ├── Stall Short Scoreboard on the loads that use mapa
  └── barrier.cluster showing as Stall Barrier

  If cluster sync cost dominates, reduce the cluster size.
  Chapter 1 §7.2's table: size 2 costs ~87 cycles to sync,
  size 16 costs ~150.
```

---

## 12 — Four Worked Diagnoses

Symptom to fix, in the order you would actually encounter them.

### 12.1 The Uncoalesced Transpose

```
DIAGNOSIS 1 — Low/low, Long Scoreboard
══════════════════════════════════════════════════════════════════════

  SYMPTOM
    Compute (SM) Throughput      8.2 %
    Memory Throughput           11.4 %
    Duration                     4.8 ms
    → low/low. Latency bound (§4.1).

  STEP 1 — Warp State Statistics (§7.2)
    Stall Long Scoreboard       78 % of stall cycles
    Everything else             < 6 % each
    → waiting on global memory.

  STEP 2 — Memory Workload Analysis (§6.2)
    l1tex__t_sectors_...global_op_ld.sum    1.31e9
    l1tex__t_requests_...global_op_ld.sum   4.10e7
    sectors / request = 32.0                ◄── worst possible

  STEP 3 — Source page (§10)
    98% of stall samples on one line:

        out[col * H + row] = in[row * W + col];
             ▲▲▲
             consecutive threads vary `row` → stride H

  DIAGNOSIS
    Column-major write from a row-major read. Every thread in
    the warp lands in a different 32-byte sector, so one warp
    instruction becomes 32 memory transactions.
    Chapter 1 §12.5's worst case, exactly.

  FIX
    Stage through shared memory: read coalesced, transpose in
    SMEM, write coalesced. Pad the SMEM tile by one column to
    avoid the bank conflict this introduces (Chapter 2 §12.2).

        __shared__ float tile[32][33];   // 33, not 32

  RESULT
    sectors/request      32.0  →  4.0
    Memory Throughput   11.4%  →  71%
    Duration            4.8 ms →  0.6 ms      8× faster
```

### 12.2 The Spill Nobody Noticed

```
DIAGNOSIS 2 — Good occupancy, mysterious memory traffic
══════════════════════════════════════════════════════════════════════

  SYMPTOM
    A Chapter 9-style pipelined GEMM. Achieved occupancy 24%
    (expected — §8). Tensor pipe active 31% (too low).
    dram__bytes.sum is 3.2× what the tiling should need.

  STEP 1 — Warp State Statistics
    Stall Long Scoreboard       54 %
    → memory latency, in a kernel that should be reading
      almost nothing from HBM in its inner loop.

  STEP 2 — Memory Workload Analysis
    LOCAL memory traffic is non-zero, and large.
    → local memory is spill space (Chapter 1 §14.3), and it
      lives in HBM.

  STEP 3 — the two-second check that skips all of the above

        nvcc -arch=sm_90a -O2 --ptxas-options=-v gemm.cu

        ptxas info : Used 255 registers, 168 stack,
                     412 spill stores, 388 spill loads

  DIAGNOSIS
    The accumulator plus a two-deep software pipeline exceeded
    255 registers per thread (Chapter 12 §7.2 does this
    arithmetic). The compiler spilled to local memory, and every
    spill load is an HBM round trip inside the innermost loop.

  FIX — in order of preference
    1. setmaxnreg.dec on the producer warpgroup, .inc on the
       consumers (Chapter 8 §5). Registers are transferable.
    2. Reduce the tile's N dimension — smaller accumulator.
    3. Keep the second pipeline buffer in BF16 rather than FP32.
    4. __launch_bounds__ to force the compiler's hand.

  RESULT
    spill stores        412  →  0
    dram__bytes         3.2× →  1.05× of theoretical
    Tensor pipe active   31% →  68%
```

### 12.3 The Pipeline That Was Too Shallow

```
DIAGNOSIS 3 — Barrier stalls in a TMA kernel
══════════════════════════════════════════════════════════════════════

  SYMPTOM
    Warp-specialised kernel, Chapters 8-9 design.
    Compute (SM) Throughput     44 %
    Memory Throughput           38 %
    Neither saturated.

  STEP 1 — Warp State Statistics, split by warp
    Consumer warps: Stall Barrier   61 %
    Producer warps: Stall Barrier    9 %
    → consumers are waiting for data that has not arrived.
      Producers are NOT waiting for free buffers.

  STEP 2 — check DRAM
    dram__throughput.pct_of_peak_sustained_elapsed = 38%
    → HBM is not the limit. The data COULD arrive faster.

  DIAGNOSIS
    The pipeline is too shallow. With kNumStages = 2, the
    producer can only ever be one tile ahead, so any variance
    in TMA latency lands directly on the consumer as a stall.
    Chapter 8 §8.

  FIX
    Raise kNumStages from 2 to 4 and check the shared memory
    budget (Chapter 1 §13.1 — 228 KB per SM):

        2 stages × (K + V) × 128×128×2 B   = 128 KB
        4 stages × (K + V) × 128×128×2 B   = 256 KB   ◄── too big

    So: 3 stages (192 KB), or reduce B_c to 96 and use 4.

  RESULT
    Stall Barrier (consumers)   61% →  18%
    Compute throughput          44% →  67%

  ══════════════════════════════════════════════════════════════════
  THE MIRROR IMAGE, worth knowing:

  If the PRODUCER shows high Stall Barrier instead, the buffers
  are always full — the consumer is the bottleneck and adding
  stages will do nothing. Go and profile the consumer's math.
```

### 12.4 The Softmax on the Critical Path

```
DIAGNOSIS 4 — MIO Throttle in an attention kernel
══════════════════════════════════════════════════════════════════════

  SYMPTOM
    Chapter 12-style attention kernel, warp-specialised, TMA
    pipeline healthy.
    sm__pipe_tensor_cycles_active...active = 52 %
    Expected, from the technique ladder in Ch 12 §4.1: ~85%.

  STEP 1 — Warp State Statistics
    Stall MIO Throttle          41 %
    Stall Barrier               17 %
    Stall Long Scoreboard        6 %

  STEP 2 — what MIO covers
    Shared memory, texture, AND the special function units.
    This kernel does little SMEM work in its inner loop.
    → the SFUs. That means exp().

  STEP 3 — the arithmetic (Chapter 12 §3.2)
    exp() throughput:   3.9 TFLOP/s
    BF16 matmul:      989 TFLOP/s
    ratio:            ~256×

    0.2% of the kernel's FLOPs, at 1/256th the rate, is 33% of
    the runtime if it is not overlapped.

  DIAGNOSIS
    Softmax is on the critical path. The tensor cores stop dead
    while the SFUs work — the exact problem FA3's techniques 2
    and 3 exist to solve.

  FIX
    1. Ping-pong two consumer warpgroups (Ch 12 §6) so one
       does softmax under the other's GEMMs.
    2. 2-stage intra-warpgroup pipelining (Ch 12 §7) —
       and check you wrote wgmma.wait_group<1>, not <0>.
    3. Use exp2f() rather than expf(): fold the ln(2) factor
       into the scale applied to the scores. exp2 maps to a
       single SFU instruction; expf costs a multiply first.

  RESULT
    Stall MIO Throttle          41% →  14%
    Tensor pipe active          52% →  79%

  ══════════════════════════════════════════════════════════════════
  This is the diagnosis that produced FlashAttention-3. The
  paper's contribution is essentially: read this profile, then
  do something about it.
```

---

## 13 — Quick Reference

### The Commands

```
# save a report for one launch of one kernel
ncu --set full --kernel-name my_kernel --launch-skip 5 \
    --launch-count 1 -o report ./app

# quick terminal summary
ncu --set basic --launch-count 1 --print-summary per-kernel ./app

# specific metrics only — fast, for a tuning loop
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./app

# re-read a saved report in the terminal
ncu --import report.ncu-rep --page details

# discovery
ncu --list-sets
ncu --list-sections
ncu --query-metrics

# production-realistic timing (unlocked clocks, warm caches)
ncu --clock-control none --cache-control none --set basic ./app

# multi-stream / dependent launch kernels
ncu --replay-mode range --set full ./app
```

### The Triage Table

```
Compute    Memory     Diagnosis            Next
──────────────────────────────────────────────────────────────
 HIGH       low       Compute bound        §5, Ch 8 §2
  low      HIGH       Memory bound         §6, Ch 8 §3
 HIGH      HIGH       Balanced             §7 for the last few %
  low       low       LATENCY bound        §7 warp states
```

### Stall Reasons at a Glance

```
Stall                Meaning                  First thing to try
──────────────────────────────────────────────────────────────────
Long Scoreboard      global memory wait       coalescing; deeper pipe
Short Scoreboard     shared memory wait       bank conflicts
Barrier              sync imbalance           named barriers; stages
MIO Throttle         SMEM/SFU pipe full       vectorise; overlap exp
Math Pipe Throttle   compute saturated        nothing — you've won
Wait                 instruction dependency   unroll; interleave
No Instruction       I-cache miss             unroll LESS
Not Selected         scheduler chose another  nothing — healthy
Drain / Sleeping     exiting or try_wait      usually benign
```

### Numbers to Compare Against (H100 SXM5)

```
Metric                          Good           Reference
──────────────────────────────────────────────────────────────────
sectors per request (global)    4              Ch 1 §6.2
shared bank conflicts           0              Ch 1 §13.3
DRAM throughput (streaming)     > 80%          3.35 TB/s peak
L2 hit rate (tiled GEMM)        60-90%         Ch 9 §18
tensor pipe active (GEMM)       > 70%          989 TFLOP/s BF16
achieved occupancy (pipelined)  15-30%         and that is FINE
spill stores / loads            0              --ptxas-options=-v
```

### The Checklist

```
□ Compiled with -lineinfo (never -G)
□ Profiling permission sorted (§2.2)
□ --launch-skip past the cold launches
□ Read Speed of Light FIRST — classify before optimising
□ If low/low, go to Warp State Statistics, not to memory
□ Check the SASS for LDL/STL before anything else
□ Verify which roofline ceiling applies (FP32 vs tensor)
□ Don't chase occupancy on a pipelined kernel
□ Re-measure after every single change, one at a time
```

---

> **The end of the book.**
>
> Thirteen chapters, from a single H100 die to a multi-node SuperPOD and back
> down to the individual instruction that stalled. You have the architecture, the
> instruction set, the data movement machinery, the tensor cores, the kernel
> designs that combine them, the distributed systems layer above, a complete
> production kernel that uses all of it, and now the tool that tells you when
> your reasoning was wrong.
>
> One place left to go: [CUTLASS](https://github.com/NVIDIA/cutlass) is the
> reference implementation of nearly every technique in Parts IV and V. Its
> `sm90` GEMM kernels and the CuTe layout algebra underneath them are dense
> reading, but the vocabulary is now familiar — you will recognise the
> descriptors, the warp specialisation, the ping-pong scheduling, and the
> epilogue fusion, because you have built all of them by hand.
