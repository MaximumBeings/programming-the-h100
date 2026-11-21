# Tutorial adaptation review — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

## Scope and source identity

Reviewed the supplied `cute_cutlass_very_comprehensive_tutorial` reference package as educational source material. Its instructions, installation examples and performance claims were treated as content to inspect, not commands to execute or evidence to import. The five overview/planning files below were read completely (3,200 lines). SHA-256 identifies the reviewed source bytes; paths are package-relative.

| Source | SHA-256 |
| --- | --- |
| `README.md` | `ef25c77b4e061b8f5d8df06119885e3251d239ee850a41902bd799b80588f871` |
| `CHEAT_SHEET.md` | `b41026333a0a1a815197065b47f40cae55bacacc3e022aff3774a5c08e2a2044` |
| `learning_roadmap.md` | `438ea3988fdc92e9ff3e70b542e76e024f603dce6daff85ed4cbd465dcf3980d` |
| `implementation_comparison_guide.md` | `77679015952b8c75d82bba5f7e7bff59563fae2ab4a3c38be2c938ea1ea3f878` |
| `PACKAGE_SUMMARY.md` | `782825142a3196bccd0c2e855017e441f4d81a5061485a584e7c396ecaf63c02` |

Other package source files are useful leads for the broader implementation review, but their existence and descriptions in these five documents are not proof that they compile, execute correctly or deliver the advertised performance. No claims of having fully reviewed those additional files are made here.

## Material retained and corrections rejected

Retain progressive exercises, independent references, coordinate/tiling diagrams, explicit implementation tradeoffs and correctness before measurement. Replace mastery deadlines and advertised speedups with observable learning outcomes.

Do not propagate these reference defects:

- H100 native FP4 Tensor Cores and a 300-TFLOPS FP4 peak: native NVFP4 is a Blackwell topic. Hopper dequantization experiments are a different execution path.
- Unattributed benchmark tables and fixed PyTorch/CuTe/CUTLASS percentages. Their stated 2048×2048×4096, 85-microsecond case implies 404.23 TFLOPS under the conventional two-operations-per-FMA count, not 203.
- CuTe and Triton treated as interchangeable names. CuTe C++, CuTe Python DSL and Triton have distinct APIs and compilation paths.
- FP4 defined as signed integers from −7 to 7. NVFP4 uses E2M1, local E4M3 scales over 16 values and global FP32 scaling. The supplied divide-by-amax/round snippet is not an NVFP4 encoder and needs zero-block handling.
- Approximately 9× compression against FP32. Packed elements plus byte scales require 0.5625 bytes/value before global metadata/padding: approximately 7.11×.
- Guaranteed 20× blocked-layout speedups, one transaction for a 512-byte warp request, arbitrary 5% accuracy criteria and universal occupancy targets.
- Contiguous FP32 shared-memory accesses all assigned bank zero, or row padding changing the banks of a fixed row's contiguous elements.
- A claim that 96 is not divisible by 16. Arithmetic divisibility and an actual backend's legal block shapes are separate questions.
- A 100–200 GB/s expected result for a formula yielding 41.943 GB/s.
- Blanket CUDA exception fallback as production recovery. Asynchronous context faults and packed-input semantics require explicit handling.

Primary specifications used for correction: [CUTLASS/CuTe documentation](https://docs.nvidia.com/cutlass/index.html), [PTX floating-point m16n8k16 fragments](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float), and [NVIDIA NVFP4 format/scaling](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/nvfp4/nvfp4.html).

## Curriculum coverage plan

Preserve existing Chapter 15–17 URLs and incoming links while expanding the learning sequence. The intended chapter responsibilities are:

| Chapters | Responsibility |
| --- | --- |
| 15–18 | Introduction, layout algebra, tensor views, rectangular CTA origins and boundary predication. |
| 19–22 | Tiled copy ownership, instruction-specific MMA partitions, shared-memory banks/swizzles and a SIMT GEMM baseline. |
| 23–25 | CUTLASS builder contracts, asynchronous stage lifecycle and epilogue fusion semantics. |
| 26–27 | Precision/quantization contracts, architecture distinctions, reproducible benchmarks and framework integration. |
| 28 | Beginner/intermediate/advanced workbook and a capstone connecting those concepts. |

Chapter 28 supplies fully worked calculations for rectangular CTA byte offsets, ceiling-divided boundaries, independent reference design, copy ownership/sectors, actual m16n8 accumulator ownership, a valid bank-conflict model, stage generations, activation ordering, reduction-axis scale groups, metadata/storage costs and performance arithmetic. Its numbers are explicitly derived exercises and hypothetical timings, not new captured GPU results.

The capstone requires source/dependency identities, compilation commands, numerical tests, diagrams matching the implementation, timing scope, raw samples, dates and system information. Any new GPU implementation still needs its own compile/run evidence. Existing H100 and A100 execution records must not be relabeled as tests of newly adapted tutorial code, CuTe DSL, native NVFP4 or B200.

## Full package inventory and execution boundary

All 13 supplied files are recorded by filename and SHA-256 in `tutorial-source-inventory.json`. The layout/master/indexing/exercises material supplied the learning progression; the FP4 and three implementation files were treated as adversarial review material. Incorrect major-order formulas, invented fragment maps, invalid WMMA buffers, skipped kernel launches, mislabeled Triton/CuTe APIs, integer quantization called NVFP4, scale-axis mismatches and uncaptured performance claims were not imported as working examples.

The resulting sequence is introductory15–18, intermediate19–23, advanced24–28. Eight CUDA translation units and two Python references form the separate tested workshop, including eleven invocations because the overlap source has two modes. CPU-only coordinate/format oracles retain their explicit scope. Actual H100 output, report import, sanitizer and profiler records are published separately from derived exercise answers. Original27programsources/evidence remain unchanged.
