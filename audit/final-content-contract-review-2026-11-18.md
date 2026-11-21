# Final content-contract review — November 18, 2026

> Historical audit, scoped to its stated date and source revision. Later source corrections and outstanding GPU reruns are tracked in [the November 21 review repair record](../docs/review-fixes-2026-11-21.md). This note does not upgrade an old execution to a result for corrected source.

This bounded follow-up read prior audits, reviewed the 29-chapter navigation and setup, and challenged residual contracts in the original early chapters. It found explanatory defects, not a confirmed new executable defect. No remote GPU was used and no new GPU execution is claimed.

## Confirmed corrections

- P1: Chapter2 now states cluster co-scheduling within one GPC, valid DSMEM initialization/readiness and target-block lifetime. Removed a physical-neighbor guarantee, per-GPC private-L2 implication, arbitrary 87/150-cycle barrier rules and active-cluster arithmetic derived only from 18 SMs. Cluster occupancy queries and componentwise grid constraints replace that false launch model.
- P2: Chapters1/2/3/4/6/13 now distinguish register/shared-memory throughput, dependency latency, bank conflict service, compiler register budgets, dynamic redistribution and cached spills. Removed universal one-cycle/energy ratios and size-based latency claims from explanatory text models. Actual captured source comments/records remain unchanged.
- P2: Chapter3 now separates matching acquire/release publication from standalone fences and separates API host return, queuing and completion. A synchronous pageable H2D copy may return before final DMA completion; Async is not an unconditional immediate-return promise. Device synchronization is scoped to the current device/context.
- P2: Chapter4's TMA capabilities now require valid descriptors and matching consumer layouts. Multicast does not promise one physical HBM access or a cluster-size speedup; swizzling does not guarantee every consumer access is conflict-free.
- P2: Chapter9 describes ping-pong as a possible overlap schedule, not guaranteed simultaneous work or zero idle time. Fusion requires a supported precision/dataflow/resource contract; arbitrary cross-tile reductions are not automatically local epilogues.

## Preserved evidence and exact explanatory-fence exceptions

The original 27-program revalidation check passed. All 10 workshop current/captured sources remain byte-identical. No CUDA/C++/Python executable source fence, captured command/output block or raw log was changed. The following 18 pre-existing untyped/text fences intentionally changed because they taught inaccurate reference models; indexes are zero-based in Markdown fence order. The machine-readable before/after SHA256 list is `audit/final-explanatory-fence-changes-2026-11-18.json`.

- `04-cutensormap-tma-swizzling.md`, fence2: TMA Capabilities Overview → TMA capabilities — supported descriptors and instruction variants.
- `04-cutensormap-tma-swizzling.md`, fence16: Conflict-Free Access — The Ideal Case → Conflict-Free Access — The Ideal Case.
- `04-cutensormap-tma-swizzling.md`, fence17: 32 Banks — The Design Rationale → 32 banks — scalar 32-bit request model.
- `13-nsight-compute.md`, fence29: Cluster Kernels (Chapter 2 §1-2) → Cluster Kernels (Chapter 2 §1-2).
- `06-wgmma-part-1.md`, fence9: Operand Location Rules — Where Data Must Reside → Operand Location Rules — Where Data Must Reside.
- `02-clusters-inline-ptx.md`, fence2: Step 1: Kernel launch with cluster size = 2 → Cluster launch and DSMEM lifetime — conceptual sequence.
- `02-clusters-inline-ptx.md`, fence3: Cluster Size Trade-offs — Practical Decision Table → Cluster-size decisions — contracts, not latency predictions.
- `02-clusters-inline-ptx.md`, fence29: State Space Map — Physical Location vs. Software View → State Space Map — Physical Location vs. Software View.
- `02-clusters-inline-ptx.md`, fence30: Register Space — What You Need to Know → Register budget — illustrative arithmetic, not occupancy prediction.
- `02-clusters-inline-ptx.md`, fence34: Local, Constant, and Parameter State Spaces → Local, Constant, and Parameter State Spaces.
- `02-clusters-inline-ptx.md`, fence35: Data flow for thread 0: → Data flow for thread 0:.
- `02-clusters-inline-ptx.md`, fence54:   4-byte "word" at byte offset B:  word_index = B / 4 →   4-byte "word" at byte offset B:  word_index = B / 4.
- `02-clusters-inline-ptx.md`, fence74: Space           PTX Qualifier        CUDA C++           Scope    Speed → Space           PTX Qualifier        CUDA C++           Scope    Speed.
- `03-asynchronicity-mbarrier.md`, fence5: Blocking Operations — Reference → Blocking and completion — distinguish host from device.
- `03-asynchronicity-mbarrier.md`, fence6: Non-Blocking Operations — Reference → Asynchronous initiation — completion is a separate fact.
- `03-asynchronicity-mbarrier.md`, fence19: Phase Bit — How Reuse Works → Phase Bit — How Reuse Works.
- `01-h100-architecture.md`, fence21: Register File → Register File.
- `01-h100-architecture.md`, fence23: What Happens When You Run Out of Registers → What Happens When You Run Out of Registers.

## Primary contracts consulted

- [CUDA 12.8 thread-block clusters](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#thread-block-clusters).
- [Hopper cluster/occupancy guidance](https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html#thread-block-clusters).
- [PTX 8.7 memory consistency model](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model).
- [CUDA Runtime API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html).
- [Hopper tuning guide/TMA](https://docs.nvidia.com/cuda/archive/12.8.1/hopper-tuning-guide/index.html).

The previous pinned CUTLASS and numerical worked-example reviews remain applicable; this pass did not identify another required algorithm change. These checks do not certify arbitrary inputs, architectures or future implementation variants.
