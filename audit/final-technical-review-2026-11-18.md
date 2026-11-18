# Final independent technical review — 2026-11-18

This review inspected instructions, reference arithmetic and explanatory models across all 13 chapters. The independent offline review was followed by the November 18 two-H100 rerun: all four barrier corrections passed numerical checks and zero-error memcheck. All 27 current programs have verified workload records across the recorded H100 and eight-A100 hosts. The later adversarial exposition review below changed no complete program or captured evidence.

## P1 — bulk-async barrier initialization needed proxy publication

The complete programs `ex01_mbarrier_basic.cu` and `ex02_war_prevention.cu` in Chapter 3, `ex02_tma_3d.cu` in Chapter 4, and `ex01_bulk_raw.cu` in Chapter 5 initialized raw shared barriers and synchronized threads, but omitted publication of initialization to the bulk-async proxy. The initializing thread's generic-proxy write must be made visible before the async operation uses that barrier. A successful observed run does not establish this memory-ordering contract.

The current sources now issue `fence.proxy.async.shared::cta` before the initialization `__syncthreads()`. The exact archived/current SHA-256 pairs and verified states are pinned in [gpu_revalidation.json](../tools/gpu_revalidation.json). Only those four complete sources changed in the original barrier correction. The adjacent book records identify the corrected revisions and their successful November 18 GPU evidence. Raw archives were preserved. Teaching snippets in Chapters 3, 5 and 9 now include the same publication requirement.

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
- Chapters 3–5: reviewed proxy directions, completed-copy visibility, phase reuse, raw-copy alignment and descriptor edge handling. The publication defect above is covered by the successful November 18 source revalidation.
- Chapters 6–7: reviewed WGMMA fence versus completion, disjoint-register requirements for partial waits, fragment storage, swizzle assumptions and sparse metadata selectors.
- Chapters 8–9: reviewed arithmetic-intensity models, register redistribution budget and generic-proxy producer/consumer handshakes.
- Chapter 10: inspected actual first/middle/final split publication and merge paths; distinguished the single-thread teaching schedule from a general GEMM scheduler.
- Chapter 11: independently traced row-major cuBLAS transpose parameters, forward/backward matrix dimensions, DP gradient averaging and the CPU serial oracle. The reference correctly uses distinct DP inputs, full matrices, per-shard clipping and rounded FP32 weights. Final checks cover 8×2048 stored weights (including replicated copies) and 2×8 step losses; they are not a convergence or throughput claim. The computed stage-0 input gradient is unused by this two-layer example.
- Chapter 12: checked the online-softmax recurrence and the 8192×128 traffic/FLOP models. Requested K/V traffic differs from measured HBM traffic because L2 reuse depends on schedule; ideal useful-I/O is not a once-per-input hardware guarantee.
- Chapter 13: checked roofline arithmetic and the separation of illustrative diagnoses from captured profiles.

These checks did not reveal another required CUDA algorithm change. They do not prove correctness for untested dimensions, precision regimes, schedules or architectures. The four revised barrier programs now have their own successful GPU and memcheck records; earlier raw records were not relabeled as runs of the new revisions.


## Final adversarial exposition pass — November 18

Confirmed and corrected residual contradictions outside all complete tested source blocks:

- Chapter 1: replaced the claim that asynchronous copying started with Hopper with the Ampere `cp.async` versus Hopper TMA distinction. Removed the incorrect claim that ordinary loads block all independent following work, and the stale twenty-thousand-instruction comparison.
- Chapter 1: replaced universal SFU one-instruction/accuracy claims with function-specific lowering and error contracts; removed unsupported instruction-cache capacity and timing guarantees. FP64 offers greater precision, not exact arithmetic.
- Chapter 3 §15: bulk groups now describe the supported shared-to-global store direction, not an interchangeable completion mechanism for TMA loads. Corrected newest-N wait semantics, source-read versus full completion, issuer-local tracking, and notification of other source-buffer producers.
- Chapter 5 §7.2: replaced invalid global-to-shared `.bulk_group` teaching instructions with the supported shared-to-global tensor-store syntax and explicit proxy-publication prerequisites. The multicast illustration now elects one issuing thread, rather than issuing once per thread in the leader CTA.
- Chapter 6: removed physical warp-fusion/one-issuer claims; aligned pipeline summaries with load → register fence → compute → completion. Corrected the worked descriptor hexadecimal arithmetic, separated it from the unswizzled complete program, and used PTX's pattern-start-based matrix-base-offset formula.
- Chapter 7: scoped the `N/16` x4 epilogue-store count to N divisible by 16 and explained the smaller remainder composition needed for valid shapes such as N=8 or N=24.
- Home page: aligned the WGMMA pipeline summary with the corrected chapter.

Primary contracts: [Ampere asynchronous copies](https://docs.nvidia.com/cuda/archive/12.8.0/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory), [PTX bulk copy](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk), [WGMMA shared-memory descriptor](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout), [CUDA mathematical functions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix), [NVIDIA's H100 instruction-cache case study](https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/).
