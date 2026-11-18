# Final independent technical review — 2026-11-18

This review inspected actual instructions, reference arithmetic and explanatory models across all 13 chapters. It did not execute GPU code. Existing run records remain evidence for their captured source revisions; the four source corrections below require new GPU evidence.

## P1 — bulk-async barrier initialization needed proxy publication

The complete programs `ex01_mbarrier_basic.cu` and `ex02_war_prevention.cu` in Chapter 3, `ex02_tma_3d.cu` in Chapter 4, and `ex01_bulk_raw.cu` in Chapter 5 initialized raw shared barriers and synchronized threads, but omitted publication of initialization to the bulk-async proxy. The initializing thread's generic-proxy write must be made visible before the async operation uses that barrier. A successful observed run does not establish this memory-ordering contract.

The current sources now issue `fence.proxy.async.shared::cta` before the initialization `__syncthreads()`. The exact archived/current SHA-256 pairs and pending states are pinned in [pending_gpu_revalidation.json](../tools/pending_gpu_revalidation.json). Only those four complete sources changed. The adjacent book notes identify the revision and pending GPU evidence. Raw archives were preserved. Teaching snippets in Chapters 3, 5 and 9 now include the same publication requirement.

Primary source: [CUDA 12.8 Programming Guide, TMA barrier initialization](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma).

Chapter 4's `ex01_tma_basic.cu` already had the required fence. Chapter 9's `ex05_mbarrier.cu` uses ordinary producer stores, not bulk-async instructions; its mention of TMA is a suggestion comment, so it was correctly excluded from this source change.

## P2 — profiling work counts and bottleneck inference

Chapter 13's illustrative example attributed 184 GFLOP to scalar FADD/FMUL/FFMA counters, obtained a rate above the scalar FP32 roof, then inferred tensor execution. Those counters exclude tensor instructions; changing the roof cannot repair a mismatched work count. The example now starts with explicitly justified tensor work and explains that a point below the memory-side roof does not prove DRAM saturation.

Primary sources: [NVIDIA counter definitions](https://docs.nvidia.com/nsight-compute/2023.1.1/pdf/NsightComputeCli.pdf), [NVIDIA's explicit scalar/tensor counter distinction](https://forums.developer.nvidia.com/t/confusion-about-the-d-f-h-mul-add-fma-count-in-the-nsight-compute/265702/7).

Chapter 13's stale 63→4095 attention-intensity assertion was aligned with Chapter 12's separate requested-traffic (~126) and ideal useful-I/O (4096) models. The base-2 exponential explanation now uses `log2(e)=1/ln(2)` explicitly and avoids promising one instruction for every source-level `exp2f` call.

## P2 — smaller instructional inaccuracies corrected

- Chapter 2's `ld.global.cv` table incorrectly promised every load reaches DRAM. The explanation now states the documented global System Memory line-refetch scope, distinct from ordering. [PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#cache-operators).
- Chapter 6's `ldmatrix.x4` arithmetic contradicted its correct four-register output: eight 16-bit values are 128 bits, not 64. Corrected.
- Chapters 1 and 7 distinguish two retained values from an exactly-two-nonzero requirement and acknowledge TF32's separate 1:2 sparse encoding. Dense-equivalent sparse arithmetic peak is not total-kernel speedup. [PTX sparse WGMMA](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async-sp).
- Chapter 3 no longer promises that a mismatched transaction count necessarily completes at a particular intermediate byte milestone.
- Chapter 5's non-executable multicast snippet had a duplicated completion qualifier; removed. Its readiness explanation now ties cluster synchronization to initialized destination barriers rather than treating cluster scheduling itself as an initialization guarantee.
- Chapter 9's raw barrier declaration comment said 64-byte alignment beside `__align__(8)`; corrected to 8-byte alignment.

## Checks of other high-risk areas

- Chapter 1: checked dense throughput arithmetic and directional versus aggregate bandwidth units.
- Chapter 2: reviewed cluster/co-residency claims and peer-address lifetime versus synchronization.
- Chapters 3–5: reviewed proxy directions, completed-copy visibility, phase reuse, raw-copy alignment and descriptor edge handling. The publication defect above is the outstanding source revalidation work.
- Chapters 6–7: reviewed WGMMA fence versus completion, disjoint-register requirements for partial waits, fragment storage, swizzle assumptions and sparse metadata selectors.
- Chapters 8–9: reviewed arithmetic-intensity models, register redistribution budget and generic-proxy producer/consumer handshakes.
- Chapter 10: inspected actual first/middle/final split publication and merge paths; distinguished the single-thread teaching schedule from a general GEMM scheduler.
- Chapter 11: independently traced row-major cuBLAS transpose parameters, forward/backward matrix dimensions, DP gradient averaging and the CPU serial oracle. The reference correctly uses distinct DP inputs, full matrices, per-shard clipping and rounded FP32 weights. Final checks cover 8×2048 stored weights (including replicated copies) and 2×8 step losses; they are not a convergence or throughput claim. The computed stage-0 input gradient is unused by this two-layer example.
- Chapter 12: checked the online-softmax recurrence and the 8192×128 traffic/FLOP models. Requested K/V traffic differs from measured HBM traffic because L2 reuse depends on schedule; ideal useful-I/O is not a once-per-input hardware guarantee.
- Chapter 13: checked roofline arithmetic and the separation of illustrative diagnoses from captured profiles.

These checks did not reveal another required CUDA algorithm change. They do not prove correctness for untested dimensions, precision regimes, schedules or architectures. New GPU verification is required for the four revised barrier programs before their current revisions can inherit PASS status.
